import os
from setuptools import setup, find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = 'Deep Preconditioned Inverse problems.'

setup(
    name="deepprecs",
    description=descr,
    long_description=open(src('README.md')).read(),
    keywords=['inverse problems',
              'deep learning',
              'seismic'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    author='Matteo Ravasi',
    author_email='matteo.ravasi@kaust.edu.sa',
    packages=find_packages(),
    use_scm_version=dict(root='.',
                         relative_to=__file__,
                         write_to=src('deepprecs/version.py')),
    setup_requires=['setuptools_scm'],

)
