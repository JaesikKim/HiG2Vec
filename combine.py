import argparse
import torch as th
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Combine Embeddings')
    parser.add_argument('-GOonly', type=str, required=True,
                        help='GOonly Embeddings')
    parser.add_argument('-gene_embedding', type=str, required=True,
                        help='Gene Embeddings')
    opt = parser.parse_args()
    
    model1 = th.load(opt.GOonly, map_location='cpu')
    model2 = th.load(opt.gene_embedding, map_location='cpu')
    object1, embedding1 = model1['objects'], model1['embeddings'].numpy()
    object2, embedding2 = model2['objects'], model2['embeddings'].numpy()
    notinObj=[]
    notinIdx=[]
    for idx, obj in enumerate(object1):
        if obj not in object2:
            notinObj.append(obj)
            notinIdx.append(idx)
    notinEmb = embedding1[notinIdx]
    object2 += notinObj
    embedding2 = np.concatenate((embedding2, notinEmb))
    model2['objects'] = object2
    model2['embeddings'] = th.from_numpy(embedding2)
    th.save(model2, opt.gene_embedding)
    
if __name__ == '__main__':
    main()