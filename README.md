# DeepInverton
## Introduction
The DeepInverton algorithm is designed to detect DNA inversion mediated phase variation in bacterial genomes just using nucleotide sequence. It works by identifying regions flanked by inverted repeats and identifying regions using the DeepInverton model. DeepInverton should be valuable to the research community, as it enables researchers to effectively identify a list of candidate invertons from massive data, provide good targets for further exploration of invertons.
![DeepInverton](https://img-blog.csdnimg.cn/direct/ca0e9488674f463283ad183c5d4a14c3.png#pic_center)
## Installation
We recommend deploying DeepInverton using `conda`
```
# clone this repository
git clone https://github.com/HUST-NingKang-Lab/DeepInverton.git
cd DeepInverton
# configure environment using environment.yml (Time consumption of ~6s)
conda env create -f environment.yml
# activate the environment
conda activate deepinverton
```
## Usage
 Perform a search and identification of nucleotide sequences, including assembled contigs, genomics or single sequences.
```
python deepinverton.py -f input_sequence.fna -o result_dir_path --model /deepinverton/model/DeepInverton.pth -x prefix_filename  -g 15 85 -p
```
## Quick Start
All you need to get started just are nucleotide sequences (in fasta format). Then, you can search for the invertons using DeepInverton.

To test DeepInverton, you can use the example files (genomic.txt)

example:
```
python deepinverton.py -f /deepinverton/example/genomic.fna -o /deepinverton/example/result --model /deepinverton/model/DeepInverton.pth -x genomic  -g 15 85 -p
```
If successful, the output will be in /deepinverton/expample/result/ with three files, including genomic_ir.txt, genomic_inverton.txt and genomic_ir_possibility.txt.(Time consumption of ~4min)


## Parameters deepInverton.py

 - `-f  --fasta`
 	input nucleotide sequence file in fasta format 	
 -  `-o  --outdir`
  where output files should be written. default is current working directory.
 -  `-x  --prefix`
 	base name for output files
 -  `-d   --model`
    the path of DeepInverton model
 -  `-e   --einv`
 	einverted parameters, if unspecified run with DeepInverton default pipeline
 -  `-m  --mismatch`
 	max number of mismatches allowed between IR pairs, used with `-einv` (default:3)
 -  `-r  --IRsize`
 	max size of the inverted repeats, used with `-einv` (default:50)
 -  `-g  --gcrange`
  the minimum and maximum value of GC ratio
 -  `-p  --polymer`
 	Eliminate homopolymer inverted repeats
## Result file
The DeepInverton program will generate three output files, including prefix_ir.txt, prefix_inverton.txt and prefix_ir_possibility.txt.

| filename        | description          |
|:-----------:| :-------------:|
| prefix_ir.txt | output table with inverted repeats coordinates |
|prefix_inverton.txt| output table with invertons coordinates |
|prefix_ir_possibility.txt| output table with inverted repeats possibility of invertons |

 - prefix_ir.txt, prefix_inverton.txt

 | Column name | Explanation                                                   |
|-------------|---------------------------------------------------------------|
 ID    | The sequence name of inverted repeat combined with Scaffold, pos A, pos B, pos C and pos D
 Scaffold    | The sequence name where the inverted repeat is detected
 PosA       | The start coordinate of the first inverted repeat (0-based)
 PosB       | The end coordinate of the first inverted repeat (1-based)
 PosC       | The start coordinate of the second inverted repeat (0-based)
 PosD       | The end coordinate of the second inverted repeat (1-based)
 IrA       | The sequence of the first inverted repeat
Mid       | The sequence of the invertible promoter
  IrB       | The sequence of the second inverted repeat 

- prefix_ir_possibility.txt

 | Column   name | Explanation                                                   |
|-------------|---------------------------------------------------------------|
 ID    | The sequence name of inverted repeat combined with Scaffold, pos A, pos B, pos C and pos D
positive       | The  inverton possibility of the inverted repeat 
negative       | The non-inverton possibility of the inverted repeat 
