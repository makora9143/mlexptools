#!/usr/bin/env python
from os import path

from setuptools import find_packages, setup

AUTHOR = "M. Kawano"
NAME = "mlexptools"
PACKAGES = find_packages()

REQUIREMENTS_FILE = "requirements.txt"
REQUIREMENTS_PATH = path.join(path.abspath(__file__), REQUIREMENTS_FILE)
with open(REQUIREMENTS_FILE) as f:
    requirements = f.read().splitlines()

setup(
    name=NAME,
    version='0.0.1',
    description='Reusable code snippets',
    author=AUTHOR,
    author_email='kawano@weblab.t.u-tokyo.acjp',
    url='https://github.com/makora9143/mlexptools',
    install_requires=requirements,
    packages=PACKAGES,
)
