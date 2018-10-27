#!/bin/bash

# Script for Pypi release
# 0. Make sure you are on git tag
# 1. Run the script
# 2. Upload sdist

set -e

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
cd $script_dir

TAG=$(git describe --exact-match --tags HEAD)

VERSION=${TAG/v/}

WAVENET_VOCODER_BUILD_VERSION=$VERSION python setup.py develop sdist
echo "*** Ready to release! wavenet_vocoder $TAG ***"
echo "Please make sure that release verion is correct."
cat wavenet_vocoder/version.py
echo "Please run the following command manually:"
echo twine upload dist/wavenet_vocoder-${VERSION}.tar.gz --repository-url https://upload.pypi.org/legacy/
