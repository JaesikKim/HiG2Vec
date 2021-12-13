#!/bin/sh
#
# This source code is partially modified for the application to HiG2Vec.
# The original source code of Poincare Embedding can be found in  https://github.com/facebookresearch/poincare-embeddings

python3 embedGO.py \
       -dim 200 \
       -lr 0.3 \
       -epochs 1000 \
       -negs 50 \
       -batchsize 50 \
       -burnin 20 \
       -gpu 0 \
       -manifold poincare \
       -dset data/example_of_GO.tsv \
       -checkpoint result/GOonly_200dim.pth \
       -fresh \
       -sparse

python3 embedGene.py \
       -dim 200 \
       -lr 1 \
       -epochs 1000 \
       -negs 50 \
       -batchsize 50 \
       -burnin 20 \
       -gpu 0 \
       -manifold poincare \
       -dset data/example_of_goa_human.tsv \
       -pretrain result/GOonly_200dim.pth \
       -checkpoint result/hig2vec_200dim.pth \
       -sym \
       -fresh \
       -sparse \
       -finetune
       
python3 combine.py \
       -GOonly result/GOonly_200dim.pth \
       -gene_embedding result/hig2vec_200dim.pth