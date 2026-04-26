# MANE : Manifold Aligned Neighbor Embedding

We introduce a neighbor embedding framework for manifold alignment. We demonstrate the efficacy of the framework using a manifold-aligned version of the uniform manifold approximation and projection algorithm. We show that our algorithm can learn an aligned manifold that is visually competitive to embedding of the whole dataset.

![mane embedding](/images/FMNIST_embedding.png)
Two-dimensional embedding of Fashion-MNIST data. (Left) UMAP embedding of 60,000 points. (Right) Top row: embedding of $D^{(1)}$ and bottom row: embedding of $D^{(2)}$ for the individual UMAP, aligned UMAP, and MANE. Individual UMAPs naturally cannot align the manifolds which can be seen from misalignment of the large cluster consisting of images of ankle boot, sandal and sneaker in the two embeddings. Aligned UMAP and MANE show very good alignment.

![Embedding of Shared Points](/images/D0D0_FMNIST.png)
Shared data points from the two-dimensional embedding of the Fashion-MNIST dataset of the figure above. (Left) Individual UMAPs of sets $D^{(1)}$ and $D^{(2)}$ show that the shared information is not aligned between two datasets. (Middle) Aligned UMAP shows close alignment between the shared points. (Right) MANE shows best alignment between the shared points.

## Install

```
git clone https://github.com/tariqul-islam/mane
cd mane
pip install .
```

## Usage

```
from mane import mane_2set

emb_comm, emb1, emb2 = mane_2set(comm1, comm2, data1, data2)

#here comm1 (Nxd1) and comm2 (Nxd2) are data with correspondence 
# data1 (N1xd1) is additional data from the space of comm1
# data2 (N2xd2) is additional data from the space of comm2

#or when additional data is not available

emb_comm, _, _ = mane_2set(comm1, comm2)
```
## Citation

If you use the code, please consider citing our paper:
```
@inproceedings{islam2022manifold,
  title={Manifold-aligned Neighbor Embedding},
  author={Islam, Mohammad Tariqul and Fleischer, Jason W},
  booktitle={ICLR 2022 Workshop on Geometrical and Topological Representation Learning}
}
```
