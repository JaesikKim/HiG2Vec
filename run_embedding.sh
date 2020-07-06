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
       -dim 200 \
       -lr 1 \
       -epochs 20000 \
       -negs 50 \
       -batchsize 20000 \
       -burnin 20 \
       -gpu 2 \
       -manifold poincare \
       -dset data/GO_closure.tsv \
       -checkpoint result/GOonly_200_20000eph.pth \
       -fresh \
       -sparse

#python3 embedGene.py \
#       -dim 200 \
#       -lr 0.3 \
#       -epochs 1000 \
#       -negs 50 \
#       -batchsize 20000 \
#       -burnin 20 \
#       -gpu 2 \
#       -manifold poincare \
#       -dset data/goa_human.tsv \
#       -pretrain result/GOonly_200_ori.pth \
#       -checkpoint result/hig2vec_200_ori5.pth \
#       -fresh \
#       -sparse \
#       -finetune
       
#python3 combine.py \
#       -GOonly result/GOonly_200_ori.pth \
#       -gene_embedding result/hig2vec_200_ori5.pth
