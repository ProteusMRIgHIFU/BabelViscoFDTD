#!/usr/bin/env python3
"""
Standalone PrepareKernels script — invoked by CMake at configure time.

Merges GPU kernel headers (GPU_KERNELS.h / GPU_KERNELS2D.h) into single-file
OpenCL sources and copies the indexing headers into the BabelViscoFDTD package
directory so they are available at runtime and are included in the wheel.
"""
import os
import sys
from pathlib import Path
from shutil import copyfile


def PrepareKernels(src_dir=None, pkg_dir=None):
    root = Path(__file__).parent
    if src_dir is None:
        src_dir = root / "src"
    else:
        src_dir = Path(src_dir)
    if pkg_dir is None:
        pkg_dir = root / "BabelViscoFDTD"
    else:
        pkg_dir = Path(pkg_dir)

    for kernel_h, out_c in [
        ("GPU_KERNELS.h",   "_gpu_kernel.c"),
        ("GPU_KERNELS2D.h", "_gpu_kernel2D.c"),
    ]:
        with open(src_dir / kernel_h, "r") as f:
            lines = f.readlines()
        with open(pkg_dir / out_c, "w") as f:
            for line in lines:
                if "#include" not in line:
                    f.write(line)
                else:
                    inc_file = line.split('"')[1]
                    with open(src_dir / inc_file, "r") as g:
                        f.writelines(g.readlines())

    copyfile(src_dir / "Indexing.h",   pkg_dir / "_indexing.h")
    copyfile(src_dir / "Indexing2D.h", pkg_dir / "_indexing2D.h")
    print("PrepareKernels: done")


if __name__ == "__main__":
    PrepareKernels()
