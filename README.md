# DeepInverton
## Introduction
The DeepInverton is a deep learning framework to identify a inverton with nucleotide sequence.
![DeepInverton](https://img-blog.csdnimg.cn/2972dda365974ff8aed36cdf1e4a784a.png#pic_center)

## Installation

```
# clone this repository
git clone https://github.com/HUST-NingKang-Lab/DeepInverton.git
cd DeepInverton

# configure environment using environment.yaml
conda install mamba -n base -c conda-forge -y
mamba env create -f environment.yml
conda activate DeepInverton
```
## Usage
- Run search and indentify for assembled contigs or genomics

```
DeepInverton.py -m
```
- Run search and indentify for small sequence shorter than 300 nts

```
DeePInverton.py -m
```
## Parameters DeepInverton.py
### Input Data Options
