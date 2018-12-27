import os
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

version = {}
with open("morphops/_version.py") as version_file:
    exec(version_file.read(), version)

setup(
    name='morphops',
    version=version['__version__'],
    description='Geometric morphometrics operations in python',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Vai Patel',
    author_email='vai.patel@gmail.com',
    url='https://github.com/vaipatel/morphops',
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers = [
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering'
    ],
    install_requires= ['numpy'],
    include_package_data=True
)