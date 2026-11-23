import os
from glob import glob

import sys
if sys.version_info > (3, 2):
    unicode=str
    def bCheckIfStr(v):
        return type(v) is str
    def cStr(v):
        if type(v) is bytes:
            return v.decode("utf-8")
        else:
            return v
else:
    def bCheckIfStr(v):
        return (type(v) is str or type(v) is unicode)
    def cStr(v):
        return v
    
def bCheckdNumpyStr(v):
    if numpy.__version__ < "2.0.0":
        return type(v) is numpy.string_ or type(v) is numpy.unicode_ 
    else:
        return type(v) is numpy.str_

'''
Generic BackEnd to save hdf5 files using h5py, it seems more efficient than Pytables for "very-general-purpose saving". For better designed groups and datasets, pytables is way better

'''
import numpy
import h5py
import hdf5plugin 

# Blosc splits each chunk across this many threads (read at every (de)compression
# call, and inherited by child processes). Users can override it by setting the
# variable beforehand.
os.environ.setdefault('BLOSC_NTHREADS', str(min(8, os.cpu_count() or 1)))

def _SlabChunks(v, target=16*2**20):
    '''Chunks made of whole slabs along the first axis, of about `target` bytes.
    Large chunks are what lets Blosc use multiple threads; h5py's automatic
    chunks (~100 kB) leave it single-threaded'''
    row = max(1, v.nbytes // v.shape[0])
    return (max(1, min(v.shape[0], target // row)),) + v.shape[1:]
from pydicom.uid import UID
from collections import OrderedDict

def ProcType(k,v,f,compatibility,complevel,group):
    nametype=type(v).__name__
    if group is None:
        group=f['/']
    if  type(v) in [dict,OrderedDict,H5LazyDict]:
        if isinstance(k,str):
            indexType='str'
        elif isinstance(k,unicode):
            if sys.version_info <= (3, 2):
                indexType='unicode'
            else:
                indexType='str'
        elif isinstance(k,int) or isinstance(k,numpy.int32) or isinstance(k,numpy.int16) or isinstance(k,numpy.int64):
            indexType='int'
        elif isinstance(k,float) or isinstance(k,numpy.float32) or isinstance(k,numpy.float64):
            indexType='float'
        else:
            raise TypeError('The type of key ' +str(k)+ ' is not supported ( ' + str(type(k)) + ' )')
        stk = k if (isinstance(k,str) or isinstance(k,unicode)) else str(k)
        newgroup=group.create_group(stk)
        newgroup.attrs["type"]="OrderedDict" if type(v) is OrderedDict else "dict"
        newgroup.attrs["type_key"]=indexType
        SaveToH5py(v,f,compatibility,complevel,newgroup)
    elif type(v) in [list,H5LazyList]:
        newgroup=group.create_group(k)
        newgroup.attrs["type"]="list";
        for n in range(len(v)):
            vlist=v[n]
            itemname = "item_%d"%n
            ProcType(itemname,vlist,f,compatibility,complevel,newgroup)
    elif type(v)  is tuple:
        nameList = k
        newgroup=group.create_group(k)
        newgroup.attrs["type"]="tuple"
        for n in range(len(v)):
            vlist=v[n]
            itemname = "item_%d"%n
            ProcType(itemname,vlist,f,compatibility,complevel,newgroup)
    elif type(v)  is  numpy.ndarray :
        ##we can apply compression rules for larger arrays
        if v.nbytes >2**10: # if array takes more 1024 bytes
            if compatibility=="lzf":
                ds=group.create_dataset(k, data=v,compression="lzf",chunks=True,shuffle=True) #this is very simple but accessible compressor
            elif compatibility=="gzip":
                ds=group.create_dataset(k, data=v,compression="gzip",compression_opts=complevel,chunks=True,shuffle=True) #this gives a good compression and it is Matlab compatible
            elif compatibility=="blosc":
                ds=group.create_dataset(k, data=v, **hdf5plugin.Blosc(clevel=complevel),chunks=_SlabChunks(v)) #this gives a good compression and it is blasting fast
        else:
            ds=group.create_dataset(k, data=v)
        ds.attrs["type"]="ndarray"
    elif  bCheckIfStr(v) or bCheckdNumpyStr(v):
        ds=group.create_dataset(k, data=v)
        ds.attrs["type"]=str(type(v))
    elif numpy.isscalar(v):
        ds=group.create_dataset(k, data=v)
        ds.attrs["type"]="scalar"
    elif v is None:
        ds=group.create_dataset(k, data=0)
        ds.attrs["type"]="None"
    elif nametype == 'UID':
        sUID=str(v)
        ds=group.create_dataset(k, data=sUID)
        ds.attrs["type"]=nametype
    else:
        raise TypeError( "Datatype not handled:" + nametype)


def SaveToH5py(MyDict,f,compatibility="blosc",complevel=9,group=None):
    if bCheckIfStr(f):
        #stream readers of this file in this process keep their data and release the handle
        CloseAllH5Streams(bLoadPending=True,fname=f)
        fileobj=h5py.File(f, "w")
    elif type(f) is h5py._hl.files.File:
        fileobj=f
    else:
        raise TypeError( "only string or h5py.file objects are accepted for 'f' object")

    if type(MyDict) not in [dict,OrderedDict,H5LazyDict]:
        raise TypeError("Only dictionaries are supported to be saved")
    for (k, v )in MyDict.items():
        ProcType(k,v,fileobj,compatibility,complevel,group)
    if bCheckIfStr(f):
        fileobj.close() #if we receive a string, that means that we open and close the file

def ReadFromH5py(f,group=None,typeattr=None):
    if bCheckIfStr(f):
        fileobj=h5py.File(f, "r")
    elif type(f) is h5py._hl.files.File:
        fileobj=f
    else:
        raise TypeError( "only string or h5py.file objects are accepted for 'f' object")
    if group is None:
        group=fileobj
        MyDict={}
    else:
        if typeattr=='OrderedDict':
            MyDict=OrderedDict()
        else:
            MyDict={}
    for (namevar,val) in group.items():

        NameList=None
        typeattr=cStr(val.attrs["type"])
        if type(val) is h5py._hl.group.Group:
            ValGroup=ReadFromH5py(fileobj,val)
            if typeattr=="list" or  typeattr=="tuple" :
                NameList=namevar
            if NameList is not None:
                ListVal=[]
                for n in range(len(ValGroup)):
                    itemname = "item_%d"%n
                    ListVal.append(ValGroup[itemname])
                if typeattr=="list":
                    MyDict[NameList] = ListVal
                else:
                    MyDict[NameList] = tuple(ListVal)
            else:

                if "type_key" in val.attrs:
                    typekey= cStr(val.attrs["type_key"])
                    if typekey=='int':
                        snamevar=int(namevar)
                    elif typekey=='float':
                        snamevar=float(namevar)
                    elif typekey=='str':
                        snamevar=namevar
                    elif typekey=='unicode':
                        snamevar=namevar
                    else:
                        raise TypeError( "the type of dictionary key is not supported " + namevar + ' (' + typekey + ')')
                else:
                    snamevar=namevar
                MyDict[snamevar] = ValGroup
        elif typeattr == "None":
             MyDict[namevar]=None
        elif typeattr == "UID":
            MyDict[namevar]=UID(val[()])
        else:
            if cStr(val.attrs["type"])=="scalar":
                if type(val[()]) is numpy.int32 or type(val[()]) is numpy.int64:
                    MyDict[namevar]=int(val[()])
                else:
                    MyDict[namevar] = val[()]
            elif cStr(val.attrs["type"])=="<type 'str'>":
                MyDict[namevar] = cStr(val[()])
            else:
                MyDict[namevar] = val[()]
    if bCheckIfStr(f):
        fileobj.close()
    return MyDict


'''
Lazy (streaming) reader
=======================

ReadFromH5pyStream(f) returns the same structure as ReadFromH5py(f), but no
dataset is read from disk until its entry is first accessed. Once read, the
value is cached in the container and never read again.

  - dict groups  -> H5LazyDict (subclass of dict)
  - list groups  -> H5LazyList (subclass of list)
  - tuple groups -> plain tuple (immutable, so its items are read when the
                    tuple itself is accessed; nested dict/list stay lazy)

Values are restored exactly as ReadFromH5py does (same types, same
bytes-vs-str behaviour, same key conversion and ordering). The HDF5 file is
only ever opened read-only. Containers can be modified in memory as usual
(d['x']*=1e3, d[k]=..., del d[k]); that never touches the file.

The file handle is shared by all containers coming from the same call:
  d.close()       release the OS handle; a later access to a not-yet-loaded
                  entry re-opens the file, provided it did not change on disk
                  (otherwise H5StaleFileError is raised)
  d.load_all()    read every pending entry in place (recursive)
  d.to_dict()     plain dict/list/tuple copy, identical to ReadFromH5py output
  with ReadFromH5pyStream(fn) as d: ...   closes the handle on exit

SaveToH5py on a path that has open stream readers in this process first
loads their pending entries (so they keep the old data, as ReadFromH5py
would) and closes them. For writers in OTHER processes call
CloseAllH5Streams() before launching them.
'''
import threading
import weakref
from collections.abc import ItemsView, ValuesView

class H5StaleFileError(RuntimeError):
    pass

_OpenStreamSources = weakref.WeakSet()

class _H5Pending(object):
    '''Placeholder stored in lazy containers for a not-yet-read HDF5 object'''
    __slots__ = ('path',)
    def __init__(self, path):
        self.path = path
    def __repr__(self):
        return '<not loaded: %s>' % self.path

def _FileSignature(fname):
    st = os.stat(fname)
    return (st.st_ino, st.st_size, st.st_mtime_ns)

class _H5StreamSource(object):
    '''Shared, thread-safe, read-only handle to the HDF5 file behind lazy containers'''
    def __init__(self, f):
        self.lock = threading.RLock()
        if bCheckIfStr(f):
            self.fname = os.path.realpath(f)
            self._file = h5py.File(self.fname, "r")
            self._signature = _FileSignature(self.fname)
            self.owned = True
            _OpenStreamSources.add(self)
        elif type(f) is h5py._hl.files.File:
            self.fname = os.path.realpath(f.filename)
            self._file = f
            self._signature = None
            self.owned = False #caller manages the file object
        else:
            raise TypeError( "only string or h5py.file objects are accepted for 'f' object")
        self.containers = weakref.WeakValueDictionary() #id->lazy container alive, used by release()

    @property
    def file(self):
        with self.lock:
            if self._file is None or not self._file:
                if not self.owned:
                    raise RuntimeError("The h5py.File object passed to ReadFromH5pyStream was closed by the caller")
                if _FileSignature(self.fname) != self._signature:
                    raise H5StaleFileError("%s was modified after it was opened by ReadFromH5pyStream; "
                                           "call load_all() before the file is rewritten" % self.fname)
                self._file = h5py.File(self.fname, "r")
            return self._file

    def close(self):
        with self.lock:
            if self.owned and self._file is not None:
                if self._file:
                    self._file.close()
                self._file = None

    def release(self):
        '''load every pending entry, then close'''
        with self.lock:
            for c in list(self.containers.values()):
                c.load_all()
            self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

def _ReadDatasetLikeReadFromH5py(val):
    #mirrors the dataset branch of ReadFromH5py exactly
    typeattr=cStr(val.attrs["type"])
    if typeattr == "None":
        return None
    elif typeattr == "UID":
        return UID(val[()])
    elif typeattr=="scalar":
        v=val[()]
        if type(v) is numpy.int32 or type(v) is numpy.int64:
            return int(v)
        else:
            return v
    elif typeattr=="<type 'str'>":
        return cStr(val[()])
    else:
        return val[()]

def _ResolveH5Object(src, path):
    with src.lock:
        obj = src.file[path]
        if type(obj) is h5py._hl.group.Group:
            typeattr=cStr(obj.attrs["type"])
            if typeattr=="list":
                return H5LazyList(src, obj)
            elif typeattr=="tuple":
                tmp = H5LazyList(src, obj)
                return tuple(tmp)
            else:
                return H5LazyDict(src, obj)
        else:
            return _ReadDatasetLikeReadFromH5py(obj)

def _DictKeyFromGroup(namevar, obj):
    #mirrors key conversion of ReadFromH5py
    if type(obj) is h5py._hl.group.Group and cStr(obj.attrs["type"]) not in ["list","tuple"] \
            and "type_key" in obj.attrs:
        typekey= cStr(obj.attrs["type_key"])
        if typekey=='int':
            return int(namevar)
        elif typekey=='float':
            return float(namevar)
        elif typekey in ['str','unicode']:
            return namevar
        else:
            raise TypeError( "the type of dictionary key is not supported " + namevar + ' (' + typekey + ')')
    return namevar

def _Materialize(v):
    if isinstance(v, H5LazyDict):
        return {k: _Materialize(v[k]) for k in dict.keys(v)}
    if isinstance(v, H5LazyList):
        return [_Materialize(x) for x in v]
    if type(v) is dict:
        return {k: _Materialize(x) for k, x in v.items()}
    if type(v) is list:
        return [_Materialize(x) for x in v]
    if type(v) is tuple:
        return tuple(_Materialize(x) for x in v)
    return v

class _H5LazyMixin(object):
    def close(self):
        self._src.close()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self._src.close()
        return False
    def to_dict(self):
        '''plain (non-lazy) deep structure, identical to ReadFromH5py output'''
        return _Materialize(self)
    def __copy__(self):
        return self.copy()
    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.to_dict(), memo)
    def __reduce_ex__(self, protocol):
        #pickle as plain containers (h5py handles can't be pickled)
        return (_Materialize, (self.to_dict(),))

class H5LazyDict(_H5LazyMixin, dict):
    '''dict whose values are read from HDF5 on first access and then cached'''
    def __init__(self, src, group):
        dict.__init__(self)
        self._src = src
        base = group.name.rstrip('/')
        for namevar in group.keys(): #same (alphanumeric) order as ReadFromH5py
            path = base + '/' + namevar
            dict.__setitem__(self, _DictKeyFromGroup(namevar, group[namevar]), _H5Pending(path))
        src.containers[id(self)] = self

    def _get(self, k, v):
        if type(v) is _H5Pending:
            with self._src.lock:
                v = dict.get(self, k, v) #another thread may have loaded it
                if type(v) is _H5Pending:
                    v = _ResolveH5Object(self._src, v.path)
                    dict.__setitem__(self, k, v)
        return v

    def __getitem__(self, k):
        return self._get(k, dict.__getitem__(self, k))
    def get(self, k, default=None):
        if dict.__contains__(self, k):
            return self[k]
        return default
    def setdefault(self, k, default=None):
        if dict.__contains__(self, k):
            return self[k]
        dict.__setitem__(self, k, default)
        return default
    def pop(self, k, *default):
        if dict.__contains__(self, k):
            v = self[k]
            dict.__delitem__(self, k)
            return v
        return dict.pop(self, k, *default)
    def popitem(self):
        k = next(reversed(dict.keys(self)))
        return k, self.pop(k)
    # __iter__ is overridden so that dict(d), {**d}, d2.update(d) go through
    # keys() + __getitem__ instead of copying the raw placeholders
    def __iter__(self):
        return iter(dict.keys(self))
    def items(self):
        return ItemsView(self)
    def values(self):
        return ValuesView(self)
    def copy(self):
        return {k: self[k] for k in dict.keys(self)}
    def _loaded(self):
        return {k: self[k] for k in dict.keys(self)}
    def load_all(self):
        for v in self.values():
            if isinstance(v, (H5LazyDict, H5LazyList)):
                v.load_all()
            elif type(v) is tuple:
                for x in v:
                    if isinstance(x, (H5LazyDict, H5LazyList)):
                        x.load_all()
        return self
    def __eq__(self, other):
        return self._loaded() == other
    def __ne__(self, other):
        return not self == other
    __hash__ = None
    def __or__(self, other):
        r = self.copy(); r.update(other); return r
    def __ror__(self, other):
        r = dict(other); r.update(self); return r
    def __repr__(self):
        return 'H5LazyDict(' + dict.__repr__(self) + ')'

class H5LazyList(_H5LazyMixin, list):
    '''list whose items are read from HDF5 on first access and then cached'''
    def __init__(self, src, group):
        list.__init__(self)
        self._src = src
        base = group.name.rstrip('/')
        list.extend(self, [_H5Pending(base + '/item_%d' % n) for n in range(len(group))])
        src.containers[id(self)] = self

    def _get(self, n):
        v = list.__getitem__(self, n)
        if type(v) is _H5Pending:
            with self._src.lock:
                v = list.__getitem__(self, n)
                if type(v) is _H5Pending:
                    v = _ResolveH5Object(self._src, v.path)
                    list.__setitem__(self, n, v)
        return v

    def __getitem__(self, n):
        if isinstance(n, slice):
            return [self._get(i) for i in range(*n.indices(len(self)))]
        return self._get(n)
    def __iter__(self):
        n = 0
        while n < len(self):
            yield self._get(n)
            n += 1
    def __reversed__(self):
        for n in range(len(self) - 1, -1, -1):
            yield self._get(n)
    def _loaded(self):
        return [self._get(n) for n in range(len(self))]
    def load_all(self):
        for v in self:
            if isinstance(v, (H5LazyDict, H5LazyList)):
                v.load_all()
            elif type(v) is tuple:
                for x in v:
                    if isinstance(x, (H5LazyDict, H5LazyList)):
                        x.load_all()
        return self
    def pop(self, n=-1):
        v = self._get(n)
        list.pop(self, n)
        return v
    def copy(self):
        return self._loaded()
    def __contains__(self, x):
        return x in self._loaded()
    def index(self, *args):
        return self._loaded().index(*args)
    def count(self, x):
        return self._loaded().count(x)
    def remove(self, x):
        self.load_all(); list.remove(self, x)
    def sort(self, *args, **kw):
        self.load_all(); list.sort(self, *args, **kw)
    def __add__(self, other):
        return self._loaded() + list(other)
    def __radd__(self, other):
        return list(other) + self._loaded()
    def __iadd__(self, other):
        list.extend(self, list(other)); return self
    def extend(self, other):
        list.extend(self, list(other))
    def __mul__(self, n):
        return self._loaded() * n
    __rmul__ = __mul__
    def __imul__(self, n):
        self.load_all(); return list.__imul__(self, n)
    def __eq__(self, other):
        return self._loaded() == other
    def __ne__(self, other):
        return self._loaded() != other
    def __lt__(self, other):
        return self._loaded() < other
    def __le__(self, other):
        return self._loaded() <= other
    def __gt__(self, other):
        return self._loaded() > other
    def __ge__(self, other):
        return self._loaded() >= other
    __hash__ = None
    def __repr__(self):
        return 'H5LazyList(' + list.__repr__(self) + ')'

def ReadFromH5pyStream(f):
    '''
    Lazy equivalent of ReadFromH5py(f). See notes above.
    f: file name (handle owned by the returned object) or an open h5py.File
    (caller keeps ownership and must keep it open while entries are pending)
    '''
    src = _H5StreamSource(f)
    with src.lock:
        return H5LazyDict(src, src.file['/'])

def CloseAllH5Streams(bLoadPending=False, fname=None):
    '''
    Release the file handles of stream readers in this process (optionally only
    those of fname). Use before another process rewrites the files. With
    bLoadPending=True pending entries are read first, so the readers stay fully
    usable; otherwise later access to a pending entry re-opens the file, or
    raises H5StaleFileError if it changed on disk.
    '''
    target = None if fname is None else os.path.realpath(fname)
    for src in list(_OpenStreamSources):
        if target is None or src.fname == target:
            if bLoadPending:
                src.release()
            else:
                src.close()
