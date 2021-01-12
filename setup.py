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
    description='Geometric Morphometrics operations in Python',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Vaibhav Patel',
    author_email='vai.pateln@gmail.com',
    url='https://github.com/vaipatel/morphops',
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers = [
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering'
    ],
    python_requires='>=3.5.1',
    setup_requires= ['numpy >= 1.13.3'],
    install_requires= ['numpy >= 1.13.3','scipy >= 1.3.3'],
    include_package_data=True
)