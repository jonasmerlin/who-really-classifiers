#!/usr/bin/env bash

wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki.tar.gz
tar -xzf wiki.tar.gz
mkdir ../../datasets/
mv wiki/ ../../datasets/
rm wiki.tar.gz
