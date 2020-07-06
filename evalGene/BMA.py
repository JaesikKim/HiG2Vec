import sys
sys.path.append("..")
from hype.poincare import PoincareManifold
import pandas as pd
import numpy as np
import torch as th
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import auc, roc_curve

def poincare_sim(a,b):
    return (1/(1+PoincareManifold().distance(a, b))).numpy()

def cosine_sim(a,b):
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()

def l2_sim(a,b):
    return 1/(1+np.linalg.norm(a - b, axis=0, ord=2))

    
def load_data(samples, objects, embeddings, g_go_dic, distfn):
    yhat_ls = []
    y_ls = []
    for i in range(len(samples)):
        g1 = samples.iloc[i,0]
        g2 = samples.iloc[i,1]
        if g1 in g_go_dic.keys() and g2 in g_go_dic.keys():
            gos1 = [go for go in g_go_dic[g1] if go in objects]
            gos2 = [go for go in g_go_dic[g2] if go in objects]
            if len(gos1) == 0 or len(gos2) == 0:
                continue
            sim = np.zeros((len(gos1), len(gos2)))
            for j,go1 in enumerate(gos1):
                for k,go2 in enumerate(gos2):
                    go1i = objects.index(go1)
                    go2i = objects.index(go2)
                    emb1, emb2 = embeddings[go1i], embeddings[go2i]
                    sim[j,k] = distfn(emb1, emb2)                    
            yhat_ls.append(np.concatenate((np.amax(sim, axis=0), np.amax(sim, axis=1))).mean())
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
        model = torch.load(opt.model, map_location="cpu")
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

    goa = pd.read_csv(opt.goa, header=None, sep="\t").values
    g_go_dic = {}
    for i in range(len(goa)):
        g = goa[i,0]
        go = goa[i,1]
        if g in g_go_dic.keys():
            ls_tmp = g_go_dic[g].copy()
            ls_tmp.append(go)
            g_go_dic[g] = np.unique(ls_tmp).tolist()
        else:
            g_go_dic[g] = [go]

    yhat, y = load_data(data, objects, embeddings, g_go_dic, distfn)
    fpr, tpr, thresholds = roc_curve(y, yhat)
    print(auc(fpr, tpr))

   
if __name__ == '__main__':
    main()

