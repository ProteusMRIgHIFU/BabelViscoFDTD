#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <arrayobject.h>
#include <stdlib.h>
/*
    * This struct allows the dynamic configuration of the allocator funcs
    * of the `page_data_allocator`. It is provided here for
    * demonstration purposes, as a valid `ctx` use-case scenario.
    */
typedef struct {
    void *(*malloc)(size_t);
    void *(*calloc)(size_t, size_t);
    void *(*realloc)(void *, size_t);
    void (*free)(void *);
} SecretDataAllocatorFuncs;
NPY_NO_EXPORT void *
page_alloc(void *ctx, size_t sz) {
    char *real ;
    int res = posix_memalign((void**)&real, PAGE_SIZE, sz);
    if (res!=0)
    {
        return NULL;
    }
    return (void *)(real);
}
NPY_NO_EXPORT void *
page_zero(void *ctx, size_t sz, size_t cnt) {
    char *real ;
    int res = posix_memalign((void**)&real, PAGE_SIZE, sz);
    if (res!=0)
    {
        return NULL;
    }
    memset(real, 0, sz );
    return (void *)(real);
}
NPY_NO_EXPORT void
page_free(void *ctx, void * p, npy_uintp sz) {
    SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *)ctx;
    if (p == NULL) {
        return ;
    }
    char *real = (char *)p;
    funcs->free(real);
}
NPY_NO_EXPORT void *
page_realloc(void *ctx, void * p, npy_uintp sz) {
    SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *)ctx;
    if (p != NULL) {
        char *real = (char *)funcs->realloc(p, sz);
        if (real == NULL) {
            return NULL;
        }
        return (void *)(real);
    }
    else {
        char *real;
        int res = posix_memalign((void**)&real, PAGE_SIZE, sz);
        if (res!=0)
        {
            return NULL;
        }
        return (void *)(real);
    }
}
/* As an example, we use the standard {m|c|re}alloc/free funcs. */
static SecretDataAllocatorFuncs secret_data_handler_ctx = {
    malloc,
    calloc,
    realloc,
    free
};
static PyDataMem_Handler secret_data_handler = {
    "page_data_allocator",
    1,
    {
        &secret_data_handler_ctx, /* ctx */
        page_alloc,              /* malloc */
        page_zero,               /* calloc */
        page_realloc,            /* realloc */
        page_free                /* free */
    }
};
void warn_on_free(void *capsule) {
    PyErr_WarnEx(PyExc_UserWarning, "in warn_on_free", 1);
    void * obj = PyCapsule_GetPointer(capsule,
                                        PyCapsule_GetName(capsule));
    free(obj);
};

static PyObject* get_default_policy(PyObject *self, PyObject *args) {
    Py_INCREF(PyDataMem_DefaultHandler);
    return PyDataMem_DefaultHandler;
}

static PyObject* set_page_data_policy(PyObject *self, PyObject *args) {
    PyObject *secret_data =
                 PyCapsule_New(&secret_data_handler, "mem_handler", NULL);
             if (secret_data == NULL) {
                 return NULL;
             }
             PyObject *old = PyDataMem_SetHandler(secret_data);
             Py_DECREF(secret_data);
             return old;
}

static PyObject* set_old_policy(PyObject *self, PyObject *args) {
    PyObject *old;
    if (args != NULL && PyCapsule_CheckExact(args)) {
        old = PyDataMem_SetHandler(args);
    }
    else {
        old = PyDataMem_SetHandler(NULL);
    }
    return old;
}     
static PyObject* get_array(PyObject *self, PyObject *args) {
    char *buf = (char *)malloc(20);
    npy_intp dims[1];
    dims[0] = 20;
    PyArray_Descr *descr =  PyArray_DescrNewFromType(NPY_UINT8);
    return PyArray_NewFromDescr(&PyArray_Type, descr, 1, dims, NULL,
                                        buf, NPY_ARRAY_WRITEABLE, NULL);  
}    
static PyObject* set_own(PyObject *self, PyObject *args) {
   if (!PyArray_Check(args)) {
        PyErr_SetString(PyExc_ValueError,
                        "need an ndarray");
        return NULL;
    }
    PyArray_ENABLEFLAGS((PyArrayObject*)args, NPY_ARRAY_OWNDATA);
    // Maybe try this too?
    // PyArray_BASE(PyArrayObject *)args) = NULL;
    Py_RETURN_NONE; 
}    
static PyObject* get_array_with_base(PyObject *self, PyObject *args) {
   char *buf = (char *)malloc(20);
    npy_intp dims[1];
    dims[0] = 20;
    PyArray_Descr *descr =  PyArray_DescrNewFromType(NPY_UINT8);
    PyObject *arr = PyArray_NewFromDescr(&PyArray_Type, descr, 1, dims,
                                            NULL, buf,
                                            NPY_ARRAY_WRITEABLE, NULL);
    if (arr == NULL) return NULL;
    PyObject *obj = PyCapsule_New(buf, "buf capsule",
                                    (PyCapsule_Destructor)&warn_on_free);
    if (obj == NULL) {
        Py_DECREF(arr);
        return NULL;
    }
    if (PyArray_SetBaseObject((PyArrayObject *)arr, obj) < 0) {
        Py_DECREF(arr);
        Py_DECREF(obj);
        return NULL;
    }
    return arr;
}    
        
static PyMethodDef page_methods[] = { 
    {   
        "get_default_policy", get_default_policy, METH_NOARGS,
        "get_default_policy."
    },  
    {   
        "set_page_data_policy", set_page_data_policy, METH_NOARGS,
        "set_page_data_policy"
    },  
    {   
        "set_old_policy", set_old_policy, METH_O,
        "set_old_policy"
    },  
    {   
        "get_array", get_array, METH_NOARGS,
        "get_array"
    },  
    {   
        "set_own", set_own, METH_O,
        "set_own"
    },  
    {   
        "get_array_with_base", get_array_with_base, METH_NOARGS,
        "get_array_with_base"
    },  

    {NULL, NULL, 0, NULL}
};

struct module_state {
    PyObject *error;
};


#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int myextension_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int myextension_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

#define STR_MAIN_NAME_1 "_page_memory"
#define INIT_MAIN_NAME PyInit__page_memory



static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        STR_MAIN_NAME_1,
        NULL,
        sizeof(struct module_state),
        page_methods,
        NULL,
        myextension_traverse,
        myextension_clear,
        NULL
};

#define INITERROR {}

PyMODINIT_FUNC INIT_MAIN_NAME (){
	    PyObject *module = PyModule_Create(&moduledef);
	  import_array();  // Must be present for NumPy. Called first after above line.
		if (module == NULL)
	        INITERROR;
	    struct module_state *st = GETSTATE(module);

	    st->error = PyErr_NewException(STR_MAIN_NAME_1 ".Error", NULL, NULL);
	    if (st->error == NULL) {
	        Py_DECREF(module);
	        INITERROR;
	    }

	    return module;
	}


