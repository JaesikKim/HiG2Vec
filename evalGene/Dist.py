import torch as th
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from hype.poincare import PoincareManifold
from sklearn.metrics import auc, roc_curve

def poincare_sim(a,b):
    return (1/(1+PoincareManifold().distance(a, b))).numpy()

def cosine_sim(a,b):
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()

def l2_sim(a,b):
    return 1/(1+np.linalg.norm(a - b, axis=0, ord=2))

def load_data(samples, objects, embeddings, distfn):
    yhat_ls = []
    y_ls = []
    for i in range(len(samples)):
        g1 = samples.iloc[i,0]
        g2 = samples.iloc[i,1]
        if g1 in objects and g2 in objects:
            g1i = objects.index(g1)
            g2i = objects.index(g2)
            emb1, emb2 = embeddings[g1i], embeddings[g2i]
            yhat_ls.append(distfn(emb1, emb2))
            y_ls.append(samples.iloc[i,2])
    return np.array(yhat_ls), np.array(y_ls)
    
def main():
    parser = argparse.ArgumentParser(description='Predict protein interaction')
    parser.add_argument('-model', help='Embedding model (.pth)', type=str)
    parser.add_argument('-dset', help='protein-protein interactions', type=str)
    parser.add_argument('-goa', help='Gene Ontology Annotation', type=str)
    parser.add_argument('-distfn', choices=['euclidean', 'poincare', 'cosine'], default='poincare', help='Distance function', type=str)
    opt = parser.parse_args()

    # load embeddings
    if opt.model[-3:] == "pth":
        model = th.load(opt.model, map_location="cpu")
        objects, embeddings = model['objects'], model['embeddings'].cpu()

    else:
        model = np.load(opt.model, allow_pickle=True).item()
        objects, embeddings = model['objects'], model['embeddings']

    # dataset processing
    print("... load data ...")
    if opt.dset[-3:] == "tsv":
        data = pd.read_csv(opt.dset, sep="\t")
    else:
        data = pd.read_csv(opt.dset)

    if opt.distfn == "poincare":
        distfn = poincare_sim
    elif opt.distfn == "cosine":
        distfn = cosine_sim
    elif opt.distfn == "euclidean":
        distfn = l2_sim

    yhat, y = load_data(data, objects, embeddings, distfn)
    fpr, tpr, thresholds = roc_curve(y, yhat)
    print(auc(fpr, tpr))

    
if __name__ == '__main__':
    main()
    

