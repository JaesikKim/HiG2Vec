#!/bin/sh
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This source code is partially modified for the application to HiG2Vec.
# The original source code of Poincare Embedding can be found in  https://github.com/facebookresearch/poincare-embeddings

python3 embed.py \
       -dim 1000 \
       -lr 0.3 \
       -epochs 1000 \
       -negs 50 \
       -burnin 20 \
       -gpu 0 \
       -manifold poincare \
       -dset ../data/GO_closure.tsv \
       -checkpoint result/test.pth \
       -batchsize 50 \
       -fresh \
       -sparse