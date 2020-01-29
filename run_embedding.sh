#!/bin/sh
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This source code is partially modified for the application to HiG2Vec.
# The original source code of Poincare Embedding can be found in  https://github.com/facebookresearch/poincare-embeddings

python3 embedGO.py \
       -dim 10 \
       -lr 0.3 \
       -epochs 25 \
       -negs 50 \
       -batchsize 50 \
       -burnin 20 \
       -gpu 0 \
       -manifold poincare \
       -dset ../data/GO_closure.tsv \
       -checkpoint result/testGO.pth \
       -fresh \
       -sparse

python3 embedGene.py \
       -dim 10 \
       -lr 1 \
       -epochs 25 \
       -negs 50 \
       -batchsize 50 \
       -burnin 20 \
       -gpu 0 \
       -manifold poincare \
       -dset ../data/goa_human.tsv \
       -pretrain result/testGO.pth \
       -checkpoint result/test.pth \
       -fresh \
       -sparse \
       -finetune
       
python3 combine.py \
       -GOonly ../poincare_embeddings/result/testGO.pth \
       -gene_embedding result/test.pth