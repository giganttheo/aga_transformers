#!/usr/bin/env python

try:
	from setuptools import setup
except BaseException:
	from distutils.core import setup


setup(name='aga_transformers',
      version="0.0.1", #__version__,
      description='Arbitrary Graph Attention Transformers',
      author='Théo Gigant',
      author_email='theo.gigant@l2s.centralesupelec.fr',
      url='https://github.com/giganttheo/aga_transformers',
      packages=['aga_transformers'],
      install_requires=[
		'numpy',
	],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.9',
    ],
     )