from setuptools import setup, find_packages
import os

setup(name='quantlib',
      version='0.1',
      description='A library to explore QNNs',
      author='Matteo Spallanzani',
      author_email='spmatteo@iis.ee.ethz.ch',
      package_dir={'quantlib': os.path.curdir},
      packages=['quantlib'] + ['.'.join(['quantlib', p]) for p in find_packages(os.path.curdir)],
      install_requires=[
          'torch',
          'networkx',
          'graphviz',
      ],
     )

