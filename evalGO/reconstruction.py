import sys
sys.path.append("..")
import torch as th
import numpy as np
import pandas
import argparse
from hype.graph import load_edge_list, eval_reconstruction
from hype.euclidean import EuclideanManifold
from hype.poincare import PoincareManifold
from sklearn.metrics.pairwise import cosine_distances

MANIFOLDS = {
    'euclidean': EuclideanManifold,
    'poincare': PoincareManifold
}

def cosine_dist(a,b):
    return th.Tensor(cosine_distances(a, b)).view(-1)

def main():
    parser = argparse.ArgumentParser(description='Reconstruction error')
    parser.add_argument('-model', type=str, required=True,
                        help='Embedding model (.pth)')
    parser.add_argument('-eval', type=str, required=True,
                        help='Evaluation corpus')
    parser.add_argument('-distfn', choices=['euclidean', 'poincare', 'cosine'], default='poincare',
                        help='Distance function') 
    opt = parser.parse_args()

    model = th.load(opt.model, map_location='cpu')
    if opt.distfn == 'cosine':
        distfn = cosine_dist
    else:
        distfn = MANIFOLDS[opt.distfn]().distance
    lt = model['embeddings']
    idx, objects = pandas.factorize(model['objects'])

    
    df = pandas.read_csv(opt.eval, header=None, sep='\t', engine='c')
    df.dropna(inplace=True)
    evalidx, evalobjects = pandas.factorize(df[[0, 1]].values.reshape(-1))
    evalidx = evalidx.reshape(-1, 2).astype('int')        

    mapping = {}
    for i in range(len(evalobjects)):
        if evalobjects[i] in objects:
            j = np.where(objects == evalobjects[i])[0][0]
            mapping[i] = j
    mapped_idx = np.array([mapping[i] if i in mapping.keys() else np.nan for i in evalidx.reshape(-1)]).reshape(-1,2)
    mapped_idx = mapped_idx[~np.isnan(mapped_idx).any(axis=1)].astype('int')

    adj = {}
    for row in mapped_idx:
        x = row[0]
        y = row[1]
        if x in adj:
            adj[x].add(y)
        else:
            adj[x] = {y}

    meanrank, maprank = eval_reconstruction(adj, lt, distfn)
    print("meanrank: ", str(meanrank))
    print("maprank: ", str(maprank))
    

if __name__ == '__main__':
    main()
