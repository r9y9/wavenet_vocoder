#!/usr/bin/env python

from setuptools import setup, find_packages
import setuptools.command.develop
import setuptools.command.build_py
import os
import subprocess

version = '0.1.1'

# Adapted from https://github.com/pytorch/pytorch
cwd = os.path.dirname(os.path.abspath(__file__))
if os.getenv('WAVENET_VOCODER_BUILD_VERSION'):
    version = os.getenv('WAVENET_VOCODER_BUILD_VERSION')
else:
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
        version += '+' + sha[:7]
    except subprocess.CalledProcessError:
        pass
    except IOError:  # FileNotFoundError for python 3
        pass


class build_py(setuptools.command.build_py.build_py):

    def run(self):
        self.create_version_file()
        setuptools.command.build_py.build_py.run(self)

    @staticmethod
    def create_version_file():
        global version, cwd
        print('-- Building version ' + version)
        version_path = os.path.join(cwd, 'wavenet_vocoder', 'version.py')
        with open(version_path, 'w') as f:
            f.write("__version__ = '{}'\n".format(version))


class develop(setuptools.command.develop.develop):

    def run(self):
        build_py.create_version_file()
        setuptools.command.develop.develop.run(self)


setup(name='wavenet_vocoder',
      version=version,
      description='PyTorch implementation of WaveNet vocoder',
      packages=find_packages(),
      cmdclass={
          'build_py': build_py,
          'develop': develop,
      },
      install_requires=[
          "numpy",
          "scipy",
          "torch >= 0.4.0",
      ],
      extras_require={
          "train": [
              "docopt",
              "tqdm",
              "tensorboardX",
              "nnmnkwii >= 0.0.11",
              "keras",
              "scikit-learn",
              "lws",
          ],
          "test": [
              "nose",
              "pysptk >= 0.1.9",
              "librosa",
              "matplotlib",
              "tqdm",
              "nnmnkwii >= 0.0.11",
          ],
      })
