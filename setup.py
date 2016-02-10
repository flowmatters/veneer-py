RELEASE = True

from setuptools import setup, find_packages
import sys, os

classifiers = """\
Development Status :: 5 - Production/Stable
Environment :: Console
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: ISC License (ISCL)
Operating System :: OS Independent
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules
"""

version = '0.1'

setup(
        name='veneer-py',
        version=version,
        description="Support for scripting eWater Source models through the Veneer (RESTful HTTP) plugin",
        long_description=read("README.md"),
        classifiers=filter(None, classifiers.split("\n")),
        keywords='hydrology ewater veneer scripting http rest',
        author='Joel Rahman',
        author_email='joel@flowmatters.com.au',
        url='https://github.com/flowmatters/veneer-py',
        #download_url = "http://cheeseshop.python.org/packages/source/p/Puppy/Puppy-%s.tar.gz" % version,
        license='ISC',
        py_modules=['veneer'],
        include_package_data=True,
        zip_safe=True,
        test_suite = 'nose.collector',
        install_requires=[
            'numpy',
            'pandas'
        ],
        extras_require={
            'test': ['nose'],
        },
)
