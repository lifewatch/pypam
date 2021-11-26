import setuptools

setuptools.setup(
    name='lifewatch-pypam',
    version='0.1.5',
    description='Facilitate acoustic processing from underwater acoustic recorders',
    author='Clea Parcerisas',
    author_email='cleap@vliz.be',
    url="https://github.com/lifewatch/pypam.git",
    license='',
    test_suite='tests',
    tests_require=['lifewatch-pypam'],
    packages=setuptools.find_packages(),
    install_requires=['pyhydrophone', 'noisereduce', 'xarray', 'seaborn', 'numpy==1.19.0'],
    package_data={"lifewatch-pypam": ["tests/test_data/*.*"]}
)
