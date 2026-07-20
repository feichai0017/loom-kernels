"""Build Loom's optional Torch CUDA custom operator in place.

This is intentionally separate from the CPU-only wheel. Run it inside the same
CUDA/PyTorch environment that hosts vLLM.
"""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


PYTHON_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PYTHON_ROOT.parent


setup(
    name="loom-attention-cuda",
    version="2.0.0a1",
    ext_modules=[
        CUDAExtension(
            name="loom_attention._cuda_ops",
            sources=[
                str(PYTHON_ROOT / "csrc" / "loom_cuda_ops.cpp"),
                str(REPOSITORY_ROOT / "cuda" / "src" / "attention_kernels.cu"),
            ],
            include_dirs=[str(REPOSITORY_ROOT / "cuda" / "include")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    package_dir={"": str(PYTHON_ROOT / "src")},
    packages=["loom_attention"],
)
