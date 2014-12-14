#!/usr/bin/env python3

from setuptools import setup, find_packages


setup(
    name="qwfilter",
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy"],
    
    # metadata for upload to PyPI
    author="Dimas Abreu Dutra",
    author_email="dimasadutra@gmail.com",
    description='"QW" filtering and smoothing library.',
    license="MIT",
    url="http://github.com/dimasad/qwfilter",
)
