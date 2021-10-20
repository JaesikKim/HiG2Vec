Code implementation of HiG2Vec: hierarchical representations of Gene Ontology and genes in the Poincaré ball
[[Paper](https://academic.oup.com/bioinformatics/article/37/18/2971/6184857)]

### Installation
Simply clone this repository via

```bash
$ git clone https://github.com/JaesikKim/HiG2Vec.git
$ cd HiG2Vec
$ python setup.py build_ext --inplace 
```

### Corpus
Gene Ontology and Gene Ontology Annotation are available in official website (http://geneontology.org/)

### Preprocessing
Transitive closure of GO
```bash
$ data/transitive_closure.py -dset data/GO.tsv
```

### Train
```bash
$ ./run_embedding.sh
```

### Evaluation
```bash
$ python evalGO/link_prediction.py -dset evalGO/GO_samples.txt -model result/hig2vec.pth -distfn poincare
```

```bash
$ python evalGO/reconstruction.py -model result/hig2vec.pth -eval data/GO_closure.tsv -distfn poincare
```

```bash
$ python evalGO/level_prediction.py -dset evalGO/level_samples.txt -model result/hig2vec.pth -fout evalGO/level_output.txt 
```

```bash
$ python evalGene/binary_prediction_NN.py -dset evalGene/STRING_samples_binary.csv -model result/hig2vec.pth -fout evalGene/binary_output.txt
```

```bash
$ python evalGene/binary_prediction_NN.py -dset evalGene/STRING_samples_binary.csv -model result/hig2vec.pth -fout evalGene/binary_output.txt
```

```bash
$ python multilabel_prediction_NN.py -dset evalGene/STRING_samples_multilabel.csv -model result/hig2vec.pth
```

```bash
$ python evalGene/score_prediction_NN.py -dset evalGene/STRING_samples_score.csv -model result/hig2vec.pth -fout evalGene/score_output.txt
```

### GO and gene embeddings
[[Download Link](https://drive.google.com/drive/folders/1WIjFSGh9E3z-PIXNOxbwjmG8EJc2j4XT?usp=sharing)] for HiG2Vec 200 dim and 1000 dim (GOonly, Human, Mouse, and Yeast)


Python code for usage
```python
import torch

model = torch.load("HiG2Vec.pth", map_location="cpu")
objects, embeddings = model['objects'], model['embeddings']
```

### Dependencies
- python 3 with numpy
- pytorch >= 1.6.0
- scikit-Learn
- pandas
- tqdm
- cpython

### Citation
```
@article{10.1093/bioinformatics/btab193, 
  author = {Kim, Jaesik and Kim, Dokyoon and Sohn, Kyung-Ah},
  title = "{HiG2Vec: hierarchical representations of Gene Ontology and genes in the Poincaré ball}",
  journal = {Bioinformatics},
  year = {2021}
}
```

** License
Software code is under MIT license, and the pre-trained HiG2Vec (GOonly, Human, Mouse, and Yeast) are under CC0 license
