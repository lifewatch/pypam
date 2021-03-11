from setuptools import setup, find_packages

setup(
    name='pypam',
    version='0.0.1',
    packages=find_packages(),
    test_suite='tests',
    tests_require=['pypam']
)
