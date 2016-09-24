#! /usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import print_function

import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "parseme-support",
    version = "0.2",
    author = "Maarten van Gompel, Federico Sangati",
    author_email = "proycon@anaproy.nl",
    description = ("Support scripts for annotation with FLAT in the PARSEME project"),
    license = "GPL",
    keywords = "",
    url = "https://github.com/proycon/parseme-support",
    packages=['tsv2folia'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Operating System :: POSIX",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    entry_points = {
        'console_scripts': [
            'tsv2folia = tsv2folia.tsv2folia:main',
        ]
    },
    #include_package_data=True,
    install_requires=['pynlpl >= 1.0']
)
