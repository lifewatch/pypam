import setuptools

setuptools.setup(
    name='pypam',
    version='0.1',
    description='Facilitate acoustic processing from underwater acoustic recorders',
    author='Clea Parcerisas',
    author_email='cleap@vliz.be',
    url="https://github.com/lifewatch/pypam.git",
    license='',
    test_suite='tests',
    tests_require=['pypam'],
    packages=setuptools.find_packages(),
    install_requires=['pandas', 'soundfile', 'numpy'],
    extras_require={
        "plotting": ["matplotlib", "seaborn"]
    },
    package_data={"pypam": ["data/*.*"]},
    include_package_data=True,
)
