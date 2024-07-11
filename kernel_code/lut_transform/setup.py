import os
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.chdir(osp.dirname(osp.abspath(__file__)))

setup(
    name='lut_transform',
    packages=find_packages(),
    include_package_data=False,
    ext_modules=[
        CUDAExtension('lut_transform', [
            'src/lut_transform.cpp',
            'src/trilinear_cpu.cpp',
            'src/trilinear_cuda.cu',
            'src/tetrahedral_cpu.cpp',
            'src/tetrahedral_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })