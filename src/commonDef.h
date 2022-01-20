#ifndef COMMON_DEF
#define COMMON_DEF
//WE put a lot of the specifics if it is a module for Matlab or Python, and if it is a X64 or a CUDA module, 99% of the low-level API for C Numpy have
//a direct counterpart for Mex functions, which simplifies greatly to write a single code.
// The use of MACROS help to write agnostic code related to the interface being served


//typedef enum {none, front, back} interface_t;

//#define CHECK_FOR_NANs

#ifdef SINGLE_PREC
#define NAME_PREC "single"
#define MEX_STR "float"
typedef float  mexType;
	#ifdef MATLAB_MEX
		#define valmexFunction mxIsSingle
		#define classOut mxSINGLE_CLASS
	#else
		#define classOut PyArray_FLOAT
		#define _numpy_type NPY_FLOAT
	#endif

#else
#define NAME_PREC "double"
#define MEX_STR "double"
typedef double mexType;
	#ifdef MATLAB_MEX
		#define valmexFunction mxIsDouble
		#define classOut mxDOUBLE_CLASS
	#else
		#define classOut PyArray_DOUBLE
		#define _numpy_type NPY_DOUBLE
	#endif
#endif

#ifdef MATLAB_MEX
#define NAME_INTERFACE "Matlab"
#else
#define NAME_INTERFACE "Python_Numpy"
#endif

#if defined(CUDA)
#define MACHINE_CODE "CUDA"
#elif defined(OPENCL)
#define MACHINE_CODE "OPENCL"
#elif defined(METAL)
#define MACHINE_CODE "METAL"
#else
#define MACHINE_CODE "Intel_X64"
#endif


#pragma message ("****************************************************************")
#pragma message ("****************************************************************")
#pragma message ("Compiling for " NAME_INTERFACE ", with " NAME_PREC " precision and using "  MACHINE_CODE " coding")
#pragma message ("****************************************************************")
#pragma message ("****************************************************************")

#ifdef MATLAB_MEX
	#define VALIDATE_FIELD_MEX_TYPE(_VarName) if (!valmexFunction(_VarName ##_mx)) \
										ERROR_STRING("The variable " #_VarName " is not the right type!");
	#define VALIDATE_FIELD_UINT32(_VarName) if(!mxIsUint32(_VarName ##_mx)) \
										ERROR_STRING("The variable " #_VarName " is not uint32");
	#define VALIDATE_FIELD_INT32(_VarName) if(!mxIsInt32(_VarName ##_mx)) \
										ERROR_STRING("The variable " #_VarName " is not int32");
	#define VALIDATE_FIELD_STRING(_VarName) if(!mxIsClass(_VarName ##_mx,"char")) \
										ERROR_STRING("The variable " #_VarName " is not char/string");
	#define GET_FIELD(_VarName) mxArray * _VarName ## _mx = mxGetField(InputStruct,0,#_VarName); \
								if (_VarName ## _mx ==NULL)\
									ERROR_STRING("Parameter " #_VarName "is missing in the input structure");
	#define GET_FIELD_GENERIC(_VarName) GET_FIELD(_VarName)
	#define ERROR_STRING(_Msg,...){mexPrintf(_Msg,##__VA_ARGS__);\
		 													   mexErrMsgTxt("aborting");}
	#define GET_DATA_PR(_VarName) (mexType *)mxGetData (_VarName ##_mx)
	#define GET_DATA(_VarName) mexType * _VarName ##_pr = GET_DATA_PR(_VarName)
	#define GET_DATA_UINT32_PR(_VarName) (unsigned int *)mxGetData(_VarName ##_mx)
	#define GET_DATA_INT32_PR(_VarName) (int *)mxGetData(_VarName ##_mx)
	#define GET_DATA_STRING(_VarName)	char * _VarName ##_pr;\
	{\
		mwSize buflen; \
		buflen = mxGetNumberOfElements(_VarName ##_mx) + 1;\
		char * _VarName ##_pr = (char *) mxCalloc(buflen, sizeof(char));\
		if (mxGetString(_VarName ##_mx, _VarName ##_pr, buflen) != 0)\
			ERROR_STRING( "Could not convert string data.");}

	#define GET_M(_VarName) mxGetM(_VarName ##_mx)
	#define GET_N(_VarName) mxGetN(_VarName ##_mx)
	#define GET_O(_VarName) mxGetDimensions(_VarName ##_mx)[2]
	#define GET_P(_VarName) mxGetDimensions(_VarName ##_mx)[3]

	#define GET_NUMBER_ELEMS(_VarName) mxGetNumberOfElements(_VarName ##_mx)

  #define PRINTF(_Msg,...)\
	{ if (INHOST(SILENT)==0)\
		 mexPrintf(_Msg,##__VA_ARGS__);\
	 }

	#define BaseArray	mxArray
  #define CREATE_ARRAY(_varName) BaseArray * _varName ##_mx = mxCreateNumericArray(ndim,dims,classOut,mxREAL);\
			if ( _varName ##_mx ==NULL)\
				ERROR_STRING("Out of memory when allocating " #_varName "_mx !!");

#else

	struct module_state {
	    PyObject *error;
	};

	#if PY_MAJOR_VERSION >= 3
	#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
	#else
	#define GETSTATE(m) (&_state)
	static struct module_state _state;
	#endif

	#define VALIDATE_FIELD_MEX_TYPE(_VarName) if (PyArray_TYPE(_VarName ##_mx)!=_numpy_type) \
												ERROR_STRING("The variable " #_VarName " is not the right type!");
	#define VALIDATE_FIELD_UINT32(_VarName) if (PyArray_TYPE(_VarName ##_mx)!=PyArray_UINT32) \
										ERROR_STRING("The variable " #_VarName " is not uint32");
	#define VALIDATE_FIELD_INT32(_VarName) if (PyArray_TYPE(_VarName ##_mx)!=PyArray_INT32) \
										ERROR_STRING("The variable " #_VarName " is not int32");

	#define GET_FIELD_GENERIC(_VarName) PyObject * _VarName ##_mx = (PyObject *)PyDict_GetItemString((PyObject*)py_argDict, #_VarName);\
									if (_VarName ## _mx ==NULL)\
											ERROR_STRING("Parameter " #_VarName "is missing in the input dictionary");

	#define GET_FIELD(_VarName) PyArrayObject * _VarName ##_mx = (PyArrayObject *)PyDict_GetItemString((PyObject*)py_argDict, #_VarName);\
									if (_VarName ## _mx ==NULL)\
											ERROR_STRING("Parameter " #_VarName "is missing in the input dictionary");
	#define ERROR_STRING(_Msg,...)  {PyErr_SetString(PyExc_ValueError, _Msg ); return 0L;}
	#define GET_DATA_PR(_VarName) (mexType *)PyArray_DATA(_VarName ##_mx)
	#define GET_DATA_UINT32_PR(_VarName) (unsigned int *)PyArray_DATA(_VarName ##_mx)
	#define GET_DATA_INT32_PR(_VarName) (int *)PyArray_DATA(_VarName ##_mx)

	#if (PY_MAJOR_VERSION ==3)
			#define GET_DATA_STRING(_VarName)\
			PyObject * _VarName ## _obj__ = PyUnicode_AsEncodedString(_VarName ##_mx, "ascii",NULL);\
			char * _VarName ##_pr = PyBytes_AS_STRING(_VarName ## _obj__);
			#define RELEASE_STRING_OBJ(_VarName) Py_XDECREF(_VarName ## _obj__);
			#define VALIDATE_FIELD_STRING(_VarName) if(!PyUnicode_Check(_VarName ##_mx))\
											ERROR_STRING("The variable " #_VarName " is not char/string");
	#else
				#define VALIDATE_FIELD_STRING(_VarName) if(!PyString_Check(_VarName ##_mx))\
												ERROR_STRING("The variable " #_VarName " is not char/string");
				#define GET_DATA_STRING(_VarName)	char * _VarName ##_pr = PyString_AsString(_VarName ##_mx);
				#define RELEASE_STRING_OBJ(_VarName) { }
	#endif

	#define GET_M(_VarName) _VarName ##_mx->dimensions[0]
	#define GET_N(_VarName) _VarName ##_mx->dimensions[1]
	#define GET_O(_VarName) _VarName ##_mx->dimensions[2]
	#define GET_P(_VarName) _VarName ##_mx->dimensions[3]
	#define GET_NUMBER_ELEMS(_VarName) PyArray_SIZE(_VarName ##_mx)
	#define PRINTF(_Msg,...)\
	{ if (INHOST(SILENT)==0)\
		 PySys_WriteStdout(_Msg,##__VA_ARGS__);\
	 }
	#define BaseArray PyArrayObject
	#define CREATE_ARRAY(_varName)  __descr = PyArray_DescrFromType(_numpy_type);\
									 BaseArray * _varName ##_mx = (PyArrayObject* )PyArray_NewFromDescr(&PyArray_Type, __descr, ndim,dims, NULL,NULL,NPY_FORTRAN,NULL);\
									if (_varName ##_mx ==NULL)\
										ERROR_STRING("Out of memory when allocating " #_varName "_mx !!");


#endif
#define GET_DATA(_VarName) mexType * _VarName ##_pr = GET_DATA_PR(_VarName)
#define GET_DATA_UINT32(_VarName) unsigned int * _VarName ##_pr =GET_DATA_UINT32_PR(_VarName)
#define GET_DATA_INT32(_VarName)  int * _VarName ##_pr =GET_DATA_INT32_PR(_VarName)
#define CREATE_ARRAY_AND_INIT(_VarName,_n1,_n2,_n3)  ndim=3;\
	dims[0]=_n1; dims[1]=_n2;  dims[2]=_n3;  \
	CREATE_ARRAY(_VarName);\
    GET_DATA(_VarName); \
     memset(_VarName ## _pr,0,(_n1)*(_n2)*(_n3)*sizeof(mexType));


#ifndef MATLAB_MEX
static PyObject* mexFunction(PyObject *self, PyObject *args);

/* #################################### GLOBALS #################################### */
/* ============================ Set up the methods table =========================== */


static PyMethodDef _FDTDStaggered_3DMethods[] = {
    {"FDTDStaggered_3D",mexFunction,METH_VARARGS},
    {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

#ifdef SINGLE_PREC
#ifdef CUDA
#define MAIN_NAME FDTDStaggered3D_with_relaxation_CUDA_single
#define STR_MAIN_NAME_1 "_FDTDStaggered3D_with_relaxation_CUDA_single"
#define INIT_MAIN_NAME PyInit__FDTDStaggered3D_with_relaxation_CUDA_single
#elif defined(OPENCL)
#define MAIN_NAME FDTDStaggered3D_with_relaxation_OPENCL_single
#define STR_MAIN_NAME_1 "_FDTDStaggered3D_with_relaxation_OPENCL_single"
#define INIT_MAIN_NAME PyInit__FDTDStaggered3D_with_relaxation_OPENCL_single
#elif defined(METAL)
#define MAIN_NAME FDTDStaggered3D_with_relaxation_METAL_single
#define STR_MAIN_NAME_1 "_FDTDStaggered3D_with_relaxation_METAL_single"
#define INIT_MAIN_NAME PyInit__FDTDStaggered3D_with_relaxation_METAL_single
#else
#define MAIN_NAME FDTDStaggered3D_with_relaxation_single
#define STR_MAIN_NAME_1 "_FDTDStaggered3D_with_relaxation_single"
#define INIT_MAIN_NAME PyInit__FDTDStaggered3D_with_relaxation_single
#endif
#else
#ifdef CUDA
#define MAIN_NAME FDTDStaggered3D_with_relaxation_CUDA_double
#define STR_MAIN_NAME_1 "_FDTDStaggered3D_with_relaxation_CUDA_double"
#define INIT_MAIN_NAME PyInit__FDTDStaggered3D_with_relaxation_CUDA_double
#elif defined(OPENCL)
#define MAIN_NAME FDTDStaggered3D_with_relaxation_OPENCL_double
#define STR_MAIN_NAME_1 "_FDTDStaggered3D_with_relaxation_OPENCL_double"
#define INIT_MAIN_NAME PyInit__FDTDStaggered3D_with_relaxation_OPENCL_double
#else
#define MAIN_NAME FDTDStaggered3D_with_relaxation_double
#define STR_MAIN_NAME_1 "_FDTDStaggered3D_with_relaxation_double"
#define INIT_MAIN_NAME PyInit__FDTDStaggered3D_with_relaxation_double
#endif
#endif

#define XSTR(x) STR(x)
#define STR(x) #x

#pragma message ("****************************************************************")
#pragma message ("****************************************************************")
#pragma message ("MAIN_NAME=" XSTR(MAIN_NAME) ", STR_MAIN_NAME_1= " STR_MAIN_NAME_1 ", INIT_MAIN_NAME= "  XSTR(INIT_MAIN_NAME))
#pragma message ("****************************************************************")
#pragma message ("****************************************************************")


#define INITERROR {}

#if PY_MAJOR_VERSION >= 3
static int myextension_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int myextension_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        STR_MAIN_NAME_1,
        NULL,
        sizeof(struct module_state),
        _FDTDStaggered_3DMethods,
        NULL,
        myextension_traverse,
        myextension_clear,
        NULL
};
#endif

PyMODINIT_FUNC INIT_MAIN_NAME (){
	#if PY_MAJOR_VERSION >= 3
	    PyObject *module = PyModule_Create(&moduledef);
	#else
	    PyObject *module= Py_InitModule(STR_MAIN_NAME_1, _FDTDStaggered_3DMethods);
	#endif
	  import_array();  // Must be present for NumPy. Called first after above line.
		if (module == NULL)
	        INITERROR;
	    struct module_state *st = GETSTATE(module);

	    st->error = PyErr_NewException(STR_MAIN_NAME_1 ".Error", NULL, NULL);
	    if (st->error == NULL) {
	        Py_DECREF(module);
	        INITERROR;
	    }

	#if PY_MAJOR_VERSION >= 3
	    return module;
	#endif
}


#endif

#include "Indexing.h"

#define LOCAL_CALLOC(_VarName,_size) mexType * _VarName ##_pr = (mexType *) calloc(_size,sizeof (mexType)*INHOST(ZoneCount));\
		if (_VarName ## _pr ==NULL) \
		ERROR_STRING("Out of memory when allocating " #_VarName);

#if defined(CUDA) || defined(OPENCL) || defined(METAL)
#include "commonDefGPU.h"
#endif


#define ASSIGN_RES(_VarName)\
{\
	mexType accum=0.0;\
	_PT _index;\
	for ( CurZone=0;CurZone<INHOST(ZoneCount);CurZone++)\
	{\
			_index = hInd_ ## _VarName(i,j,k);\
			accum+=ELDO(_VarName,_index);\
	}\
	CurZone=0;\
	_index = hInd_ ## _VarName(i,j,k);\
	ELDO(_VarName ##_res,_index)=accum/INHOST(ZoneCount);\
}

#endif
