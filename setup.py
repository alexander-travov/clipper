from setuptools import setup

install_requires = [
    'cython',
    'numpy',
]

setup_requires = [
    'cython',
    'numpy',
]

def get_numpy_include():
    import numpy
    return numpy.get_include()

def get_cythonized_modules():
    from Cython.Build import cythonize
    return cythonize('*.pyx')


setup(
    name='clipper',
    version='0.1',
    install_requires=install_requires,
    setup_requires=setup_requires,
    ext_modules=get_cythonized_modules(),
    include_dirs=[get_numpy_include()],
)
