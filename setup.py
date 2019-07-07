from setuptools import setup

descr = """skan: skeleton analysis in Python.

Inspired by the "Analyze skeletons" Fiji plugin, by Ignacio Arganda-Carreras.
"""

DISTNAME            = 'skan'
DESCRIPTION         = 'Analysis of object skeletons'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Juan Nunez-Iglesias'
MAINTAINER_EMAIL    = 'juan.n@unimelb.edu.au'
URL                 = 'https://github.com/jni/skan'
LICENSE             = 'BSD 3-clause'
DOWNLOAD_URL        = 'https://github.com/jni/skan'
VERSION             = '0.8'
PYTHON_VERSION      = (3, 6)
INST_DEPENDENCIES   = []


if __name__ == '__main__':

    setup(name=DISTNAME,
        version=VERSION,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        license=LICENSE,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],
        packages=['skan', 'skan.vendored'],
        package_data={},
        install_requires=INST_DEPENDENCIES,
        entry_points = {
            'console_scripts': ['skan-gui=skan.gui:launch']
        }
    )

