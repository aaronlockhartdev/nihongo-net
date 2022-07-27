#!/bin/bash

mkdir data
cd data
curl -O https://nlp.stanford.edu/projects/jesc/data/split.tar.gz
tar -xzf split.tar.gz