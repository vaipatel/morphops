from itertools import chain
from setuptools import setup, find_packages


with open('README.rst') as readme_file:
    readme = readme_file.read()

version = {}
with open("morphops/_version.py") as version_file:
    exec(version_file.read(), version)

extra_feature_requirements = {
    "tests": ["coverage >= 5.0", "pytest >= 5.4", "pytest-cov >= 2.8.1", "tox"],
    "docs": ["sphinx < 2", "sphinx-rtd-theme"],
}
extra_feature_requirements["dev"] = list(
    chain(*list(extra_feature_requirements.values()))
)

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
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering'
    ],
    python_requires='>=3.5.1',
    extras_require=extra_feature_requirements,
    install_requires=['numpy >= 1.13.3', 'scipy >= 1.3.3'],
    include_package_data=True
)
