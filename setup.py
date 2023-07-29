#!/usr/bin/env python

from setuptools import setup, find_packages

from importlib.machinery import SourceFileLoader

version = SourceFileLoader('wavenet_vocoder.version',
                           'wavenet_vocoder/version.py').load_module().version


setup(name='wavenet_vocoder',
      version=version,
      description='PyTorch implementation of WaveNet vocoder',
      packages=find_packages(),
      install_requires=[
          "numpy",
          "scipy",
          "torch >= 0.4.1",
          "docopt",
          "joblib",
          "tqdm",
          "tensorboardX",
          "nnmnkwii >= 0.0.11",
          "scikit-learn",
          "librosa",
      ],
      extras_require={
          "test": [
              "nose",
              "pysptk >= 0.1.9",
              "matplotlib",
          ],
      })
