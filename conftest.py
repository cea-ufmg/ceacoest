"""pytest configuration module."""

import os.path

import pytest


pytest_plugins = "ceacoest.testsupport.array_cmp"


project_root = os.path.dirname(__file__)
setup_file = os.path.join(project_root, 'setup.py')
examples_dir = os.path.join(project_root, 'examples')


def pytest_ignore_collect(path, config):
    if path == setup_file:
        return True
    if path == examples_dir:
        return True
