import setuptools


setuptools.setup(
    name='maven_iuvs',
    version='0.1.0',
    description='Utilities for Python 3 analysis of MAVEN/IUVS data.',
    url='https://github.com/lasp/maven_iuvs',
    author=('Zachariah Milby,'
            ' Kyle Connour,'
            ' Mike Chaffin,'
            ' Eryn Cangi,'
            ' and the IUVS team'),
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=[
        'astropy>=4.2',
        'cartopy>=0.17.0',
        'julian>=0.14',
        'matplotlib>=3.0.3',
        'numpy>=1.10',
        'fabric>=2.7.1',
        'invoke>=1.7.3',
        'pdoc3>=0.9.1',
        'pexpect>=4.8.0',
        'pytz>=2018.9',
        'spiceypy>=2.2.0',
        'sysrsync>=0.1.2',
        'tqdm>=4.66.1',
        'twill>=2.0.1',
        'mayavi>=4.7.2',
        'PyQt5>=5.15.2',
        'h5py>=2.10.0',
        'idl_colorbars>=1.1.2',
        'scipy>=1.11.2',
        'scikit-image>=0.21.0'
        'scikit-learn'
        'pandas'
        'statsmodels'
        'dynesty'
        'ipywidgets'
    ]
)

