'''
MetalRegressionBenchmark.py

Regression + performance harness for the GPU backends of BabelViscoFDTD.

It runs a set of scenarios that mimic how BabelBrain uses the solver (multi-material
attenuating media, ReflectorMask, particle and stress sources, RMS/Peak maps, sensors),
saves all outputs as a reference, and compares later runs against that reference.
It also reports the time spent in kernel compilation and in the time-stepping loop.

The package is imported from the folder containing this repository (not from the
installed environment), so different branches can be compared without reinstalling.
The GPU kernel files are regenerated from src/ at every run (use --no-prepare to skip),
so they always match the branch that is checked out.

The same scenarios and file format are used in every branch, so references saved with one
branch (e.g. main) can be compared with another one (e.g. a release candidate):

    git checkout main
    python PerformanceTest/MetalRegressionBenchmark.py --gpu "M1" --save-ref REF_DIR
    python PerformanceTest/MetalRegressionBenchmark.py --gpu "M1" --bench 64 160 256 --csv results.csv
    git checkout <new branch>
    python PerformanceTest/MetalRegressionBenchmark.py --gpu "M1" --compare REF_DIR
    python PerformanceTest/MetalRegressionBenchmark.py --gpu "M1" --bench 64 160 256 --csv results.csv

usage:
    python PerformanceTest/MetalRegressionBenchmark.py --list-devices                # Metal devices available
    python PerformanceTest/MetalRegressionBenchmark.py --save-ref REF_DIR             # store reference outputs
    python PerformanceTest/MetalRegressionBenchmark.py --compare REF_DIR              # compare against reference
    python PerformanceTest/MetalRegressionBenchmark.py --bench 128 192 256            # timing only, BabelBrain-like skull layer

--gpu selects the device by a substring of its name (e.g. "M1", "W6800", "Vega"); by default the first device is used.
'''
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RepoDir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RepoDir))

BACKENDS = {'CUDA': 1, 'OpenCL': 2, 'Metal': 3, 'MLX': 4}

Timings = {}


def _PrepareKernels():
    '''Regenerates the merged kernel files from src/ so they match the checked-out branch'''
    sys.path.insert(0, str(RepoDir))
    from _prepare_kernels import PrepareKernels
    PrepareKernels(RepoDir / 'src', RepoDir / 'BabelViscoFDTD')


def _EnvironmentInfo(Backend, GPU_ID):
    Info = {'platform': platform.platform(),
            'machine': platform.machine(),
            'python': platform.python_version(),
            'backend': Backend,
            'gpu_filter': GPU_ID}
    try:
        Info['cpu'] = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                     capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    try:
        Git = lambda *a: subprocess.run(['git', '-C', str(RepoDir)] + list(a),
                                        capture_output=True, text=True).stdout.strip()
        Info['branch'] = Git('rev-parse', '--abbrev-ref', 'HEAD')
        Info['commit'] = Git('rev-parse', '--short', 'HEAD')
        Info['uncommitted_changes'] = Git('status', '--porcelain', '--untracked-files=no') != ''
    except OSError:
        pass
    if Backend == 'Metal':
        try:
            import metalcomputebabel
            from importlib.metadata import version
            Info['metal_devices'] = [d.deviceName for d in metalcomputebabel.get_devices()]
            Info['metalcomputebabel'] = version('metalcomputebabel')
            Info['metalcomputebabel_batch_api'] = hasattr(metalcomputebabel.Device, 'batch_begin')
            Info['metalcomputebabel_concurrent_api'] = hasattr(metalcomputebabel.Device, 'batch_barrier')
        except Exception as e:
            Info['metalcomputebabel'] = 'unavailable (%s)' % e
    return Info


def _InstallTimers(Backend):
    '''Wraps the backend class to record compilation and execution times'''
    if Backend == 'Metal':
        from BabelViscoFDTD import StaggeredFDTD_3D_With_Relaxation_METAL as Mod
        Cls = Mod.StaggeredFDTD_3D_With_Relaxation_METAL_MetalCompute
    else:
        return
    # _CompileKernels only exists in versions that defer compilation after the buffer allocation
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
    from BabelViscoFDTD.PropagationModel import PropagationModel
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
    parser.add_argument('--gpu', default='', help='substring of the GPU name (default: first device)')
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--save-ref', metavar='DIR')
    g.add_argument('--compare', metavar='DIR')
    g.add_argument('--bench', nargs='+', type=int, metavar='N')
    g.add_argument('--list-devices', action='store_true')
    parser.add_argument('--steps', type=int, default=600, help='time steps for --bench')
    parser.add_argument('--tol', type=float, default=5e-4,
                        help='max difference relative to the peak of each array (Metal vs OpenCL differ by up to ~5e-4)')
    parser.add_argument('--csv', metavar='FILE', help='append --bench results to this CSV file')
    parser.add_argument('--no-prepare', action='store_true', help='do not regenerate the kernel files from src/')
    args = parser.parse_args()

    Info = _EnvironmentInfo(args.backend, args.gpu)
    if args.list_devices:
        for n, d in enumerate(Info.get('metal_devices', [])):
            print(n, d)
        return

    if not args.no_prepare:
        _PrepareKernels()
    print('ENVIRONMENT', json.dumps(Info))
    _InstallTimers(args.backend)

    if args.bench:
        Rows = []
        for N in args.bench:
            S = _Scenario('bench', N, N, N, NumberSteps=args.steps)
            Timings.clear()
            _, Total = RunScenario(S, args.backend, args.gpu)
            Steps = S['NumberSteps']
            Exe = Timings.get('_Execution', np.nan)
            Compile = Timings.get('_InitiateCommands', np.nan) + Timings.get('_CompileKernels', 0.0)
            print(f"BENCH N={N}^3 steps={Steps} total={Total:.2f}s compile={Compile:.2f}s "
                  f"execution={Exe:.3f}s per_step={1e3 * Exe / Steps:.3f}ms Mcells/s={N ** 3 * Steps / Exe / 1e6:.0f}")
            Rows.append({'date': time.strftime('%Y-%m-%d %H:%M:%S'), 'cpu': Info.get('cpu', ''),
                         'gpu_filter': args.gpu, 'backend': args.backend, 'branch': Info.get('branch', ''),
                         'commit': Info.get('commit', ''), 'metalcomputebabel': Info.get('metalcomputebabel', ''),
                         'N': N, 'steps': Steps, 'compile_s': round(Compile, 3), 'execution_s': round(Exe, 3),
                         'ms_per_step': round(1e3 * Exe / Steps, 3)})
        if args.csv:
            bNew = not os.path.isfile(args.csv)
            with open(args.csv, 'a', newline='') as f:
                Writer = csv.DictWriter(f, fieldnames=list(Rows[0].keys()))
                if bNew:
                    Writer.writeheader()
                Writer.writerows(Rows)
            print('Results appended to', args.csv)
        return

    Folder = Path(args.save_ref or args.compare)
    Folder.mkdir(parents=True, exist_ok=True)
    if args.save_ref:
        with open(Folder / f'environment_{args.backend}.json', 'w') as f:
            json.dump(Info, f, indent=2)
    elif (Folder / f'environment_{args.backend}.json').is_file():
        with open(Folder / f'environment_{args.backend}.json') as f:
            print('REFERENCE ENVIRONMENT', f.read().replace('\n', ' '))
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
