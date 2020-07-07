# HiG2Vec

** Installation
Simply clone this repository via

#+BEGIN_SRC sh
git clone https://github.com/JaesikKim/HiG2Vec.git
#+END_SRC

** Preprocessing
Transitive closure of GO
#+BEGIN_SRC sh
data/transitive_closure.py -dset data/GO.tsv
#+END_SRC

** Train
#+BEGIN_SRC sh
./run_embedding.sh
#+END_SRC

** Evaluation
#+BEGIN_SRC sh
python evalGO/link_prediction.py -dset evalGO/GO_samples.txt -model result/hig2vec.pth -distfn poincare
#+END_SRC

#+BEGIN_SRC sh
python evalGO/reconstruction.py -model result/hig2vec.pth -eval data/GO_closure.tsv -distfn poincare
#+END_SRC

#+BEGIN_SRC sh
python evalGO/level_prediction.py -dset evalGO/level_samples.txt -model result/hig2vec.pth -fout evalGO/level_output.txt 
#+END_SRC

#+BEGIN_SRC sh
python evalGene/binary_prediction_NN.py -dset evalGene/STRING_samples_binary.csv -model result/hig2vec.pth -fout evalGene/binary_output.txt
#+END_SRC

#+BEGIN_SRC sh
python evalGene/binary_prediction_NN.py -dset evalGene/STRING_samples_binary.csv -model result/hig2vec.pth -fout evalGene/binary_output.txt
#+END_SRC

#+BEGIN_SRC sh
python multilabel_prediction_NN.py -dset evalGene/STRING_samples_multilabel.csv -model result/hig2vec.pth
#+END_SRC

#+BEGIN_SRC sh
python evalGene/score_prediction_NN.py -dset evalGene/STRING_samples_score.csv -model result/hig2vec.pth -fout evalGene/score_output.txt
#+END_SRC

** Usage of embeddings
```python
model = torch.load("HiG2Vec.pth", map_location="cpu")
objects, embeddings = model['objects'], model['embeddings']
```

** Dependencies
- Python 3 with NumPy
- PyTorch >= 1.2.0
- Scikit-Learn
- Pandas
- tqdm
