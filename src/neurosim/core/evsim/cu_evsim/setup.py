import os
import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.2.0"
os.environ["CC"] = "g++-11"
os.environ["CXX"] = "g++-11"


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


ROOT_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "src")
include_dirs = [osp.join(ROOT_DIR, "include")]
lib_dirs = [osp.join(ROOT_DIR, "lib")]

# sources are in libdir
sources = []
for d in lib_dirs:
    sources += glob.glob(osp.join(d, "*.cpp"))
    sources += glob.glob(osp.join(d, "*.cu"))

print(bcolors.OKGREEN + "Building CUDA Extension" + bcolors.ENDC)
print(
    bcolors.WARNING
    + "Compiling for RTX5090, RTX4090, RTX4080, RTX6000 Ada, Tesla L40, L40s Ada, L4 Ada "
    + "Tesla GA10x cards, RTX Ampere, RTX 3080, RTX 3090, A3000, RTX A4000, A5000, A6000"
    + "NVIDIA A40, RTX 3060, RTX 3070, RTX 3050, RTX A10, RTX A16"
    + bcolors.ENDC
)
print(bcolors.OKBLUE + "Added Sources: ", sources, bcolors.ENDC)
print(
    bcolors.OKBLUE + "Included Headers: ",
    glob.glob(osp.join(ROOT_DIR, "include", "*.h")),
    bcolors.ENDC,
)
print(bcolors.OKBLUE + "#" * 50 + bcolors.ENDC)

setup(
    name="cu_evsim",
    version=__version__,
    ext_modules=[
        CUDAExtension(
            name="_cu_evsim_ext",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                    "-Wno-sign-compare",
                    "-fPIC",
                    "-march=native",  # Optimize for current CPU architecture
                ],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--restrict",
                    "-std=c++17",
                    "-Xcompiler",
                    "-fPIC",
                    "-Xcompiler",
                    "-march=native",  # Pass CPU optimization to host compiler
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_120,code=sm_120",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

# Make sure your nvcc path is the global nvcc path, not the conda nvcc path,
# unless you know that the conda is properly setup. The global nvcc version should
# match the torch cuda version.
