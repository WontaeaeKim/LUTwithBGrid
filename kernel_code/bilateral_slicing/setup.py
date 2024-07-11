import os
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.chdir(osp.dirname(osp.abspath(__file__)))

setup(
    name='bilateral_slicing',
    packages=find_packages(),
    include_package_data=False,
    ext_modules=[
        CUDAExtension('bilateral_slicing', [
            'src/bilateral_slicing.cpp',
            'src/trilinear_slice_cpu.cpp',
            'src/trilinear_slice_cuda.cu',
            'src/tetrahedral_slice_cpu.cpp',
            'src/tetrahedral_slice_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })