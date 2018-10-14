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
    version="0.1.dev4",
    packages=find_packages(),
    install_requires=["attrdict", "numpy", "scipy", "sym2num"],
    tests_require=["pytest"],
    
    # metadata for upload to PyPI
    author="Dimas Abreu Archanjo Dutra",
    author_email="dimasad@ufmg.br",
    description='CEA control and estimation library.',
    classifiers=CLASSIFIERS.split('\n'),
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    license="MIT",
    url="http://github.com/cea-ufmg/ceacoest",
)
