import gc

import numpy as np

from .StaggeredFDTD_3D_With_Relaxation_BASE import StaggeredFDTD_3D_With_Relaxation_BASE

import mlx.core as mx


def _CheckMLXFork():
    '''
    The MLX backend needs the spichardo/mlx fork: read-write kernel inputs (input_rw_status) and the
    flattened full-occupancy dispatch (use_optimal_threadgroups). Fast math (compile_options) is upstream.
    '''
    Doc = mx.fast.metal_kernel.__doc__ or ''
    Missing = [k for k in ['input_rw_status', 'compile_options'] if k not in Doc]
    if Missing:
        raise ImportError('BabelViscoFDTD requires the spichardo/mlx fork with ' + ', '.join(Missing) +
                          ' support in mx.fast.metal_kernel for the MLX backend')


def _DeviceName():
    Info = mx.device_info() if hasattr(mx, 'device_info') else mx.metal.device_info()
    return Info['device_name']


def ListDevices():
    return [_DeviceName()]


class StaggeredFDTD_3D_With_Relaxation_MLX(StaggeredFDTD_3D_With_Relaxation_BASE):
    '''
    MLX backend for Apple Silicon.

    MLX binds each input of a custom kernel as its own Metal buffer and has no argument buffers, while Metal
    limits a kernel to 31 buffers. The large arrays are bound directly and the small and PML arrays are
    packed into a few buffers (see _BINDINGS). The pointer of each array (k_<Name>_pr) is written in the
    kernel header with its offset inside its buffer as a compile-time constant, so the kernels access every
    array as directly as the Metal backend does through its argument buffers.

    The time loop is encoded without waiting for the GPU: each kernel receives the dummy output of the
    previous one as an input (p_TOKEN), which makes MLX keep the order of the dispatches (with a memory
    barrier between them), and the CPU waits for a batch of steps only when the next one has been encoded.
    '''
    # Buffers bound to the kernels and the arrays they hold, in binding order. A buffer with a single array
    # is that array. Arrays of a packed buffer must have the same dtype.
    _BINDINGS = [
        ('p_TABLES', ['LambdaMiuMatOverH', 'LambdaMatOverH', 'MiuMatOverH', 'TauLong', 'OneOverTauSigma',
                      'TauShear', 'InvRhoMatH', 'SourceFunctions']),
        ('p_O', ['Ox', 'Oy', 'Oz']),
        ('p_IndexSensorMap', ['IndexSensorMap']),
        ('p_SourceMap', ['SourceMap']),
        ('p_MaterialMap', ['MaterialMap']),
        ('p_PML_V', ['V_x_x', 'V_y_x', 'V_z_x', 'V_x_y', 'V_y_y', 'V_z_y', 'V_x_z', 'V_y_z', 'V_z_z']),
        ('p_PML_S', ['Sigma_x_xx', 'Sigma_y_xx', 'Sigma_z_xx', 'Sigma_x_yy', 'Sigma_y_yy', 'Sigma_z_yy',
                     'Sigma_x_zz', 'Sigma_y_zz', 'Sigma_z_zz', 'Sigma_x_xy', 'Sigma_y_xy', 'Sigma_x_xz',
                     'Sigma_z_xz', 'Sigma_y_yz', 'Sigma_z_yz']),
    ] + [('p_' + k, [k]) for k in ['Vx', 'Vy', 'Vz', 'Rxx', 'Ryy', 'Rzz', 'Rxy', 'Rxz', 'Ryz',
                                   'Sigma_xx', 'Sigma_yy', 'Sigma_zz', 'Sigma_xy', 'Sigma_xz', 'Sigma_yz',
                                   'Pressure', 'SqrAcc', 'SensorOutput']]
    # ReflectorMask is not bound: reflector cells are folded into MaterialMap (see _FoldReflectorMask)
    _NOT_BOUND = ['ReflectorMask', 'Snapshots']
    _UINT_ARRAYS = ['IndexSensorMap', 'SourceMap', 'MaterialMap', 'ReflectorMask']
    _PACK_ALIGN = 64  # entries (256 bytes) between the starts of the arrays of a packed buffer
    # Metal math mode of the kernels. "fast" is not faster (the kernels are memory bound) and moves the results
    # ~100x further from the Metal and OpenCL backends than "safe" (4.7e-4 vs 3.6e-6 of the peak, regression
    # harness), so "safe" is used.
    MathMode = "safe"

    def __init__(self, arguments):
        _CheckMLXFork()
        self.MAX_SIZE_PML = 101
        self.ZoneCount = arguments['SPP_ZONES']
        self._ArraySpecs = {}  # name -> (number of entries, dtype, host data to copy or None)
        extra_params = {"BACKEND": "METAL"}
        self.bUseSingleKernel = True
        self.StepsPerBatch = 64  # steps encoded before waiting for the previous batch
        IdsToParse = []
        for t in ['STRESS', 'PARTICLE']:
            for k in ['PML_1', 'PML_2', 'PML_3', 'PML_4', 'PML_5', 'PML_6', 'MAIN_1']:
                IdsToParse.append(f"{k}_{t}")
        IdsToParse += ['SENSORS', 'SNAPSHOT']
        self.IdsToParse = IdsToParse
        super().__init__(arguments, extra_params)

    def _PostInitScript(self, arguments, extra_params):
        print("Attempting MLX Initiation...")
        print('Selecting device: ', _DeviceName())
        self._Index64 = self._RequiresIndex64(arguments)
        SCode = []
        SCode.append("#define mexType " + extra_params['td'] + "\n")
        SCode.append("#define METAL\n")
        if self.bUseSingleKernel:
            SCode.append("#define METAL_SINGLE_KERNEL\n")
        SCode.append("#define MLX\n")
        if self._Index64:
            print('Using 64-bit indexing in GPU kernels')
            SCode.append("#define METAL_INDEX64\n")
        SCode.append("#define MAX_SIZE_PML 101\n")
        extra_params['SCode'] = SCode
        self._FoldReflectorMask(arguments)
        self.ctx = mx

    def _InitSymbol(self, IP, _NameVar, td, SCode):
        # the constants go in the kernel header, before _PT is defined, so the integer type is written explicitly
        SCode.append('//MLX_CONSTANT_START\n')
        if td in ['float', 'double']:
            res = 'constant ' + td + ' ' + _NameVar + ' = %0.9g;\n' % (IP[_NameVar])
        else:
            lType = 'unsigned long' if self._Index64 else 'unsigned int'
            res = 'constant ' + lType + ' ' + _NameVar + ' = %i;\n' % (IP[_NameVar])
        SCode.append(res)
        SCode.append('//MLX_CONSTANT_END\n')

    def _InitSymbolArray(self, IP, _NameVar, td, SCode):
        SCode.append('//MLX_CONSTANT_START\n')
        res = "constant " + td + " " + _NameVar + "_pr[%i] ={\n" % (IP[_NameVar].size)
        for n in range(IP[_NameVar].size):
            if td in ['float', 'double']:
                res += "%.9g" % (IP[_NameVar][n])
            else:
                res += "%i" % (IP[_NameVar][n])
            if n < IP[_NameVar].size - 1:
                res += ',\n'
            else:
                res += '};\n'
        SCode.append(res)
        SCode.append('//MLX_CONSTANT_END\n')

    def _ownGpuCalloc(self, Name, td, dims, ArraysGPUOp):
        if Name in self._NOT_BOUND:
            return
        print("Allocating for", Name, dims, "elements")
        self._ArraySpecs[Name] = (int(dims), np.float32 if td == "float" else np.uint32, None)

    def _CreateAndCopyFromMXVarOnGPU(self, Name, ArraysGPUOp, ArrayResCPU, flags=[]):
        if Name in self._NOT_BOUND:
            return
        Host = self._HostArray(Name, ArrayResCPU)
        print("Allocating for", Name, Host.size, "elements")
        dtype = np.uint32 if Name in self._UINT_ARRAYS else np.float32
        if Name in ['SensorOutput', 'SqrAcc']:  # outputs start at zero
            Host = None
        self._ArraySpecs[Name] = (int(ArrayResCPU[Name].size if Host is None else Host.size), dtype, Host)

    def _PreExecuteScript(self, arguments, ArraysGPUOp, outparams):
        self._outparams = outparams
        Bound = [k for _, Names in self._BINDINGS for k in Names]
        assert sorted(Bound) == sorted(self._ArraySpecs), \
            'Arrays without a binding: ' + str(set(self._ArraySpecs) ^ set(Bound))
        self.gpu_arrays = {}  # binding name -> MLX array
        self._Offsets = {}  # array name -> (binding name, offset in entries)
        for Binding, Names in self._BINDINGS:
            dtype = self._ArraySpecs[Names[0]][1]
            assert all(self._ArraySpecs[k][1] == dtype for k in Names), 'mixed dtypes in ' + Binding
            Offset = 0
            for k in Names:
                self._Offsets[k] = (Binding, Offset)
                Offset += -(-max(self._ArraySpecs[k][0], 1) // self._PACK_ALIGN) * self._PACK_ALIGN
            Hosts = [self._ArraySpecs[k][2] for k in Names]
            if all(h is None for h in Hosts):
                Buffer = mx.zeros((Offset,), dtype=mx.float32 if dtype == np.float32 else mx.uint32)
            elif len(Names) == 1:
                Buffer = mx.array(Hosts[0].reshape(-1).astype(dtype, copy=False))
            else:
                Packed = np.zeros(Offset, dtype)
                for k, h in zip(Names, Hosts):
                    if h is not None:
                        o = self._Offsets[k][1]
                        Packed[o:o + h.size] = h.reshape(-1)
                Buffer = mx.array(Packed)
                del Packed
            self.gpu_arrays[Binding] = Buffer
        mx.eval(list(self.gpu_arrays.values()))
        self._ArraySpecs = {}  # drop the references to the host inputs
        self._CompileKernels()

    def _InitiateCommands(self, SCode):
        # compilation is done by _CompileKernels at the end of _PreExecuteScript, once the offsets are known
        self._KernelSource = SCode

    def _PointerDefinitions(self):
        '''k_<Name>_pr pointer of every bound array (see kernelparamsMetal.h)'''
        Lines = []
        for k, (Binding, Offset) in self._Offsets.items():
            if Offset == 0:
                Lines.append('#define k_%s_pr (%s)\n' % (k, Binding))
            else:
                Lines.append('#define k_%s_pr (%s + %iUL)\n' % (k, Binding, Offset))
        return Lines

    def ParseAndSelectCode(self, OrigLines, Current):
        '''
        Splits the source into the header (the constants) and the body of the kernel of the block Current
        '''
        sCode = [f'#define MLX_{Current}\n']
        hCode = []
        bAdd = True
        bHeader = False
        for l in OrigLines:
            if '//MLX_BLOCK' in l:
                if '_START' in l:
                    for k in self.IdsToParse:
                        if k in l and k != Current:
                            bAdd = False
                elif "_END" in l:
                    bAdd = True
            if '//MLX_CONSTANT_START' in l:
                bHeader = True
            if bAdd and not bHeader:
                sCode.append(l)
            if bHeader:
                hCode.append(l)
            if bHeader and '//MLX_CONSTANT_END' in l:
                bHeader = False
        return ''.join(hCode), ''.join(sCode)

    def _MakeKernel(self, AllC, Block, Name):
        Header, Source = self.ParseAndSelectCode(AllC, Block)
        Header = ''.join(self._ReflectorDefinitions() + self._PointerDefinitions()) + Header
        # Every array is passed as a read-write `device` buffer. Read-only inputs would be placed in the
        # `constant` address space, which serializes the divergent reads of the material tables and maps.
        # Only the per-step constants and the ordering token are read-only.
        InputNames = ['p_CONSTANT_BUFFER_UINT', 'p_TOKEN'] + list(self.gpu_arrays.keys())
        RW = [False, False] + [True] * len(self.gpu_arrays)
        return mx.fast.metal_kernel(
            name=Name,
            input_names=InputNames,
            input_rw_status=RW,
            output_names=["dummy"],
            source=Source,
            header=Header,
            ensure_row_contiguous=False,  # a hidden copy of a read-write input would lose its updates
            compile_options={"math_mode": self.MathMode})

    def _CompileKernels(self):
        AllC = self._KernelSource
        Parts = ['MAIN_1'] if self.bUseSingleKernel else ['PML_1', 'PML_2', 'PML_3', 'PML_4', 'PML_5', 'PML_6', 'MAIN_1']
        self.AllStressKernels = {k: self._MakeKernel(AllC, f"{k}_STRESS", k + "_StressKernel") for k in Parts}
        self.AllParticleKernels = {k: self._MakeKernel(AllC, f"{k}_PARTICLE", k + "_ParticleKernel") for k in Parts}
        self.SensorsKernel = self._MakeKernel(AllC, "SENSORS", "SensorsKernel")

    def _Execution(self, arguments, ArrayResCPU, ArrayResOP):
        TimeSteps = int(arguments['TimeSteps'])
        SensorSubSampling = int(arguments['SensorSubSampling'])
        SensorStart = int(arguments['SensorStart'])
        outparams = self._outparams
        N1, N2, N3, PML = outparams['N1'], outparams['N2'], outparams['N3'], outparams['PML_Thickness']
        DimsKernel = {'PML_1': (PML, N2, N3),
                      'PML_2': (PML, N2, N3),
                      'PML_3': (N1 - PML * 2, PML, N3),
                      'PML_4': (N1 - PML * 2, PML, N3),
                      'PML_5': (N1 - PML * 2, N2 - PML * 2, PML),
                      'PML_6': (N1 - PML * 2, N2 - PML * 2, PML)}
        if self.bUseSingleKernel:
            DimsKernel['MAIN_1'] = (N1, N2, N3)
            kernels = ["MAIN_1"]
        else:
            DimsKernel['MAIN_1'] = (N1 - PML * 2, N2 - PML * 2, N3 - PML * 2)
            kernels = ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6", "MAIN_1"]
        SizeKernel = {k: int(np.prod(DimsKernel[k])) for k in kernels}
        NumberSensors = int(arguments['IndexSensorMap'].size)

        # nStep and TypeSource of every step; each step reads its row (a view, not a copy)
        StepConstants = np.zeros((TimeSteps, 2), np.uint32)
        StepConstants[:, 0] = np.arange(TimeSteps)
        StepConstants[:, 1] = int(arguments['TypeSource'])
        StepConstants = mx.array(StepConstants)
        Arrays = list(self.gpu_arrays.values())
        Token = mx.zeros((1,), dtype=mx.float32)

        def Run(Kernel, Size, Step, Token):
            return Kernel(inputs=[StepConstants[Step], Token] + Arrays,
                          grid=(Size, 1, 1),
                          threadgroup=(256, 1, 1),  # ignored with use_optimal_threadgroups
                          output_shapes=[(1,)],
                          output_dtypes=[mx.float32],
                          use_optimal_threadgroups=True)[0]

        Pending = None
        for nStep in range(TimeSteps):
            for i in kernels:
                Token = Run(self.AllStressKernels[i], SizeKernel[i], nStep, Token)
            for i in kernels:
                Token = Run(self.AllParticleKernels[i], SizeKernel[i], nStep, Token)
            if (nStep % SensorSubSampling) == 0 and (nStep // SensorSubSampling) >= SensorStart:
                Token = Run(self.SensorsKernel, NumberSensors, nStep, Token)
            if (nStep + 1) % self.StepsPerBatch == 0 or nStep == TimeSteps - 1:
                mx.async_eval(Token)
                if Pending is not None:
                    mx.eval(Pending)  # at most one batch running while the next one is encoded
                Pending = Token
        mx.eval(Token)

        def HostView(Name, Size):
            Binding, Offset = self._Offsets[Name]
            return np.asarray(self.gpu_arrays[Binding])[Offset:Offset + Size]

        for i in ['SqrAcc', 'SensorOutput']:
            ArrayResCPU[i][:] = HostView(i, ArrayResCPU[i].size).reshape(ArrayResCPU[i].shape)

        for i in ['Vx', 'Vy', 'Vz', 'Sigma_xx', 'Sigma_yy', 'Sigma_zz', 'Sigma_xy', 'Sigma_xz', 'Sigma_yz', 'Pressure']:
            sz = ArrayResCPU[i].shape
            # the SPP zone is the slowest dimension
            Buffer = HostView(i, ArrayResCPU[i].size * self.ZoneCount).reshape((self.ZoneCount,) + sz)
            ArrayResCPU[i][:, :, :] = np.sum(Buffer, axis=0) / self.ZoneCount

        del self.gpu_arrays, Arrays, StepConstants, Token, Pending
        gc.collect()


def StaggeredFDTD_3D_MLX(arguments):
    Instance = StaggeredFDTD_3D_With_Relaxation_MLX(arguments)
    Results = Instance.Results
    return Results
