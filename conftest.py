'''pytest configuration module.'''

import os.path

import pytest


project_root = os.path.dirname(__file__)
setup_file_path = os.path.join(project_root, 'setup.py')


def pytest_ignore_collect(path, config):
    if path == setup_file_path:
        return True

