'''
MetalRegressionBenchmark.py

Regression + performance harness for the GPU backends of BabelViscoFDTD.

It runs a set of scenarios that mimic how BabelBrain uses the solver (multi-material
attenuating media, ReflectorMask, particle and stress sources, RMS/Peak maps, sensors),
saves all outputs as a reference, and compares later runs against that reference.
It also reports the time spent in kernel compilation and in the time-stepping loop.

The package is imported from the folder containing this repository (not from the
installed environment), so kernel changes can be tested without reinstalling.
Remember to run `python _prepare_kernels.py` after editing files in src/.

usage:
    python PerformanceTest/MetalRegressionBenchmark.py --save-ref REF_DIR      # store reference outputs
    python PerformanceTest/MetalRegressionBenchmark.py --compare REF_DIR       # compare against reference
    python PerformanceTest/MetalRegressionBenchmark.py --bench 128 192 256     # timing only, BabelBrain-like skull layer
'''
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from BabelViscoFDTD.PropagationModel import PropagationModel

BACKENDS = {'CUDA': 1, 'OpenCL': 2, 'Metal': 3, 'MLX': 4}

Timings = {}


def _InstallTimers(Backend):
    '''Wraps the backend class to record compilation and execution times'''
    if Backend == 'Metal':
        from BabelViscoFDTD import StaggeredFDTD_3D_With_Relaxation_METAL as Mod
        Cls = Mod.StaggeredFDTD_3D_With_Relaxation_METAL_MetalCompute
    else:
        return
    for Name in ['_InitiateCommands', '_CompileKernels', '_Execution']:
        if not hasattr(Cls, Name):
            continue
        Orig = getattr(Cls, Name)
        def Wrapper(self, *args, __Orig=Orig, __Name=Name, **kwargs):
            t0 = time.time()
            Res = __Orig(self, *args, **kwargs)
            Timings[__Name] = time.time() - t0
            return Res
        setattr(Cls, Name, Wrapper)


def _Collect(Res, Prefix='', Out=None):
    '''Flattens the solver outputs into a dictionary of arrays'''
    if Out is None:
        Out = {}
    if isinstance(Res, dict):
        if 'MaterialMap' in Res:  # InputParam, not an output
            return Out
        for k, v in Res.items():
            _Collect(v, Prefix + k + '/', Out)
    elif isinstance(Res, (list, tuple)):
        for n, v in enumerate(Res):
            _Collect(v, Prefix + str(n) + '/', Out)
    elif isinstance(Res, np.ndarray):
        Out[Prefix.rstrip('/')] = Res
    return Out


def _Scenario(Name, N1, N2, N3, Frequency=500e3, NumberSteps=500):
    '''Builds the inputs of a scenario; dimensions are deliberately non-cubic'''
    rng = np.random.default_rng(1234)
    SpatialStep = 1500 / Frequency / 6
    MaterialMap = np.zeros((N1, N2, N3), np.uint32)
    if Name == 'water':
        MaterialList = np.array([[1000, 1500, 0, 0, 0]], float)
    else:
        # water + a "skull" made of 20 bone-like materials with attenuation, mimicking BabelBrain
        NMat = 20
        MaterialList = np.zeros((NMat + 1, 5))
        MaterialList[0, :] = [1000, 1500, 0, 0, 0]
        for m in range(1, NMat + 1):
            MaterialList[m, :] = [1100 + 40 * m, 1800 + 50 * m, 800 + 25 * m, 50 + 10 * m, 100 + 15 * m]
        if Name == 'bench':
            # thick layer (1/3 of the domain) with spatially coherent materials (4x4x4 blocks), closer to a real skull
            k0, k1 = N3 // 3, 2 * N3 // 3
            Blocks = rng.integers(1, NMat + 1, size=(N1 // 4 + 1, N2 // 4 + 1, (k1 - k0) // 4 + 1), dtype=np.uint32)
            MaterialMap[:, :, k0:k1] = Blocks.repeat(4, 0).repeat(4, 1).repeat(4, 2)[:N1, :N2, :k1 - k0]
        else:
            k0, k1 = N3 // 2, N3 // 2 + max(N3 // 6, 4)
            MaterialMap[:, :, k0:k1] = rng.integers(1, NMat + 1, size=(N1, N2, k1 - k0), dtype=np.uint32)

    SourceMap = np.zeros((N1, N2, N3), np.uint32)
    SourceMap[N1 // 4:3 * N1 // 4, N2 // 4:3 * N2 // 4, 14] = 1
    # second source group to exercise multiple source functions
    SourceMap[N1 // 4:N1 // 2, N2 // 4:N2 // 2, 14] = 2
    Ox = np.zeros((N1, N2, N3)); Oy = np.zeros((N1, N2, N3)); Oz = np.zeros((N1, N2, N3))
    Ox[SourceMap > 0] = 0.2; Oy[SourceMap > 0] = 0.1; Oz[SourceMap > 0] = 0.97

    SensorMap = np.zeros((N1, N2, N3), np.uint32)
    SensorMap[N1 // 2, N2 // 2, ::3] = 1
    SensorMap[::5, N2 // 3, N3 // 2 + 2] = 1

    ReflectorMask = None
    if Name != 'water':
        ReflectorMask = np.zeros((N1, N2, N3), np.uint32)
        ReflectorMask[N1 // 2 - 3:N1 // 2 + 3, N2 // 2 - 3:N2 // 2 + 3, N3 - 30:N3 - 24] = 1

    Kw = dict(SensorSubSampling=2, SensorStart=0)
    if Name == 'stress':
        Oy = np.array([1]); Oz = np.array([1])
        Kw.update(TypeSource=2, SelRMSorPeak=1, SelMapsRMSPeakList=['Pressure'], SelMapsSensorsList=['Pressure'])
    elif Name == 'skull':
        Kw.update(TypeSource=0, SelRMSorPeak=3,
                  SelMapsRMSPeakList=['Vx', 'Vy', 'Vz', 'Sigmaxx', 'Sigmayy', 'Sigmazz',
                                      'Sigmaxy', 'Sigmaxz', 'Sigmayz', 'Pressure'],
                  SelMapsSensorsList=['Vx', 'Vy', 'Vz', 'Sigmaxx', 'Sigmayy', 'Sigmazz',
                                      'Sigmaxy', 'Sigmaxz', 'Sigmayz', 'Pressure'])
    else:
        # 'water' and 'bench' record only pressure maps, as BabelBrain does
        Kw.update(TypeSource=0, SelRMSorPeak=3, SelMapsRMSPeakList=['Pressure'], SelMapsSensorsList=['Pressure'])
    return dict(MaterialMap=MaterialMap, MaterialList=MaterialList, Frequency=Frequency, SourceMap=SourceMap,
                SpatialStep=SpatialStep, NumberSteps=NumberSteps, SensorMap=SensorMap, Ox=Ox, Oy=Oy, Oz=Oz,
                ReflectorMask=ReflectorMask, Kw=Kw)


def RunScenario(S, Backend, GPU_ID):
    PModel = PropagationModel()
    MaterialList = S['MaterialList']
    # use a stable time step below the CFL limit of the fastest material
    dt = 0.4 * S['SpatialStep'] / MaterialList[:, 1].max()
    TimeVector = np.arange(S['NumberSteps']) * dt
    Pulse = np.sin(2 * np.pi * S['Frequency'] * TimeVector) * 1e-3
    SourceFunctions = np.vstack([Pulse, 0.5 * Pulse]).astype(np.float32)
    t0 = time.time()
    Res = PModel.StaggeredFDTD_3D_with_relaxation(S['MaterialMap'], MaterialList, S['Frequency'], S['SourceMap'],
                                                  SourceFunctions, S['SpatialStep'], TimeVector[-1] + dt / 2,
                                                  S['SensorMap'], Ox=S['Ox'], Oy=S['Oy'], Oz=S['Oz'], NDelta=12,
                                                  DT=dt, COMPUTING_BACKEND=BACKENDS[Backend],
                                                  DefaultGPUDeviceName=GPU_ID, SILENT=1,
                                                  ReflectorMask=S['ReflectorMask'], **S['Kw'])
    Total = time.time() - t0
    return _Collect(Res), Total


REGRESSION_SCENARIOS = [('water', 70, 60, 80), ('skull', 90, 76, 110), ('stress', 80, 96, 72)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend', default='Metal', choices=list(BACKENDS.keys()))
    parser.add_argument('--gpu', default='M', help='substring of the GPU name')
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--save-ref', metavar='DIR')
    g.add_argument('--compare', metavar='DIR')
    g.add_argument('--bench', nargs='+', type=int, metavar='N')
    parser.add_argument('--steps', type=int, default=600, help='time steps for --bench')
    parser.add_argument('--tol', type=float, default=5e-4,
                        help='max difference relative to the peak of each array (Metal vs OpenCL differ by up to ~5e-4)')
    args = parser.parse_args()

    _InstallTimers(args.backend)

    if args.bench:
        for N in args.bench:
            S = _Scenario('bench', N, N, N, NumberSteps=args.steps)
            Timings.clear()
            _, Total = RunScenario(S, args.backend, args.gpu)
            Steps = S['NumberSteps']
            Exe = Timings.get('_Execution', np.nan)
            Compile = Timings.get('_InitiateCommands', np.nan) + Timings.get('_CompileKernels', 0.0)
            print(f"BENCH N={N}^3 steps={Steps} total={Total:.2f}s compile={Compile:.2f}s "
                  f"execution={Exe:.3f}s per_step={1e3 * Exe / Steps:.3f}ms Mcells/s={N ** 3 * Steps / Exe / 1e6:.0f}")
        return

    Folder = Path(args.save_ref or args.compare)
    Folder.mkdir(parents=True, exist_ok=True)
    AllOK = True
    for Name, N1, N2, N3 in REGRESSION_SCENARIOS:
        S = _Scenario(Name, N1, N2, N3)
        Timings.clear()
        Out, Total = RunScenario(S, args.backend, args.gpu)
        FName = Folder / f'{Name}_{args.backend}.npz'
        if args.save_ref:
            np.savez_compressed(FName, **Out)
            print(f'SAVED {Name}: {len(Out)} arrays, {Total:.2f}s -> {FName}')
            continue
        Ref = np.load(FName)
        Worst = 0.0
        for k in Ref.files:
            a, b = Ref[k], Out[k]
            if a.shape != b.shape:
                print(f'  {k}: shape mismatch {a.shape} vs {b.shape}')
                AllOK = False
                continue
            Scale = np.abs(a).max()
            Diff = np.abs(a.astype(np.float64) - b).max()
            Rel = Diff / Scale if Scale > 0 else Diff
            Worst = max(Worst, Rel)
            if Diff != 0:
                print(f'  {k}: max abs diff {Diff:.3e} (relative to peak {Rel:.3e})')
        Status = 'BIT-EXACT' if Worst == 0 else f'max relative diff {Worst:.3e}'
        if Worst > args.tol:
            AllOK = False
        print(f'COMPARE {Name}: {Status}, {Total:.2f}s')
    if args.compare:
        print('RESULT:', 'PASS' if AllOK else 'FAIL')
        sys.exit(0 if AllOK else 1)


if __name__ == '__main__':
    main()
