import sys
sys.path.append("..")
from hype.poincare import PoincareManifold
from hype.euclidean import EuclideanManifold
import pandas as pd
import numpy as np
import torch as th
import argparse
from sklearn.metrics import auc, roc_curve

MANIFOLDS = {
    'euclidean': EuclideanManifold,
    'poincare': PoincareManifold
}

def main():
    parser = argparse.ArgumentParser(description='Link Prediction')
    parser.add_argument('-dset', type=str, required=True,
                        help='Link samples')
    parser.add_argument('-model', type=str, required=True,
                        help='Embedding model (.pth)')
    parser.add_argument('-distfn', choices=['euclidean', 'poincare'], default='poincare',
                        help='Distance function')    
    opt = parser.parse_args()
    
    distfn = MANIFOLDS[opt.distfn]().distance
    df = pd.read_csv(opt.dset, header=None)
    model = th.load(opt.model, map_location='cpu')
    
    objects = model['objects']
    embeddings = model['embeddings']
    
    idx_i = []
    idx_j = []
    msk = []
    for i, (go1, go2) in enumerate(df[[0,1]].values):
        if go1 in objects and go2 in objects:
            msk.append(i)
            idx_i.append(objects.index(go1))
            idx_j.append(objects.index(go2))
    yhat = (1/(1+distfn(embeddings[idx_i], embeddings[idx_j]))).tolist()
    y = df.iloc[msk,2].values.tolist()
    fpr, tpr, thresholds = roc_curve(y, yhat)
    print("AUC: "+str(auc(fpr, tpr)))
        
if __name__ == '__main__':
    main()
