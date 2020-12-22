import setuptools


setuptools.setup(
    name='maven_iuvs',
    version='0.1.0',
    description='Utilities for Python 3 analysis of MAVEN/IUVS data.',
    url='https://github.com/lasp/maven_iuvs',
    author=('Zachariah Milby,'
            ' Kyle Connour,'
            ' Mike Chaffin,'
            ' and the IUVS team'),
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=[
        'astropy>=4.2',
        'numpy>=1.10',
        'matplotlib>=3.0.3',
        'pdoc3>=0.9.1',
        'julian>=0.14',
        'sysrsync>=0.1.2',
        'paramiko>=2.6.0',
        'pytz>=2018.9',
        'spiceypy>=2.2.0',
        'pexpect>=4.8.0',
        'twill>=2.0.1'
    ]
)
