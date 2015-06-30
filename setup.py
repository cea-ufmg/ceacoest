#!/usr/bin/env python3

from setuptools import setup, find_packages

DESCRIPTION = open("README.rst", encoding="utf-8").read()

CLASSIFIERS = '''\
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3 :: Only
Topic :: Scientific/Engineering
Topic :: Software Development'''

setup(
    name="ceacoest",
    version="0.1.dev3",
    packages=find_packages(),
    install_requires=["attrdict", "numpy", "scipy", "sym2num"],
    test_requires=["pytest"],
    
    # metadata for upload to PyPI
    author="Dimas Abreu Dutra",
    author_email="dimasadutra@gmail.com",
    description='CEA control and estimation library.',
    classifiers=CLASSIFIERS.split('\n'),
    platforms=["Linux", "Unix"],
    license="MIT",
    url="http://github.com/dimasad/ceacoest",
)
