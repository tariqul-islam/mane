# MANE : Manifold Aligned Neighbor Embedding

## Install

```
git clone https://github.com/tariqul-islam/mane
cd mane
pip install .
```

Will try to add global pip install later.

## Usage

```
from mane import mane_2set

emb_comm, emb1, emb2 = mane_2set(comm1, comm2, data1, data2)

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
