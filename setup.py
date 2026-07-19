from itertools import chain
from setuptools import setup, find_packages


with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

version = {}
with open("morphops/_version.py", encoding='utf-8') as version_file:
    exec(version_file.read(), version)

extra_feature_requirements = {
    "tests": ["pytest >= 8", "pytest-cov >= 5", "tox >= 4"],
    "docs": ["sphinx >= 8", "sphinx-rtd-theme >= 3"],
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
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Topic :: Scientific/Engineering'
    ],
    python_requires='>=3.12',
    extras_require=extra_feature_requirements,
    install_requires=['numpy >= 1.26', 'scipy >= 1.12'],
    include_package_data=True
)
