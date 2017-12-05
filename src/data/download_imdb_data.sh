#!/usr/bin/env bash
# this script downloads the IMDB cropped faces dataset

GIT_ROOT=`git rev-parse --show-toplevel`

wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar
tar -xzf imdb_crop.tar
mkdir -p "$GIT_ROOT/datasets/"
mv imdb_crop/ "$GIT_ROOT/datasets/"
# rm wiki.tar