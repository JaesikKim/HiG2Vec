import pandas as pd
import argparse  

def transitive_closure(closure):
    nstep=1
    while True:
        _new_relations = set((x,w,t1,t2) for x,y,t1 in closure for q,w,t2 in closure if q == y)
        new_relations = set()
        for a, b, t1, t2 in _new_relations:
            if t1 == "is_a":
                new_relations.add((a,b,t2))
            else:
                new_relations.add((a,b,t1))
        closure_until_now = closure | new_relations
        if closure_until_now == closure:
            break
        closure = closure_until_now
        nstep += 1
        print(nstep)
        GO_closure_ = open("GO_closure_"+str(nstep)+".tsv", 'w')
        for a, b, c in closure:
            GO_closure_.write(a+"\t"+b+"\t"+c+"\n")
        GO_closure_.close()
    return closure

def main():
    parser = argparse.ArgumentParser(description='generating transitive closure')
    parser.add_argument('-dset', type=str, required=True,
                        help='Edge list of original graph')
    opt = parser.parse_args()

    GO = pd.read_csv(opt.dset, sep='\t', header=None)
    tmp = set()
    for i in range(len(GO)):
        tmp.add((GO.iloc[i,0], GO.iloc[i,1], GO.iloc[i,2]))
    GO_closure = transitive_closure(tmp)

        
if __name__ == '__main__':
    main()

