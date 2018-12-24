import os
from setuptools import setup, find_packages
from version_helper import get_version

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

version, _ = get_version()

setup(
    name='morphops',
    version=version,
    description='Geometric morphometrics operations in python',
    long_description=readme,
    author='Vai Patel',
    author_email='vai.patel@gmail.com',
    url='https://github.com/vaipatel/morphops',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers = [
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering'
    ],
    install_requires= ['numpy'],
    include_package_data=True
)