from argparse import ArgumentParser
import os
import sys
import re
import tempfile
from collections import defaultdict
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import GC
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import re
import numpy as np
import torch
from torch import nn
import math
def one_hot(sequence):
    sequence=sequence.upper()
    sequence=re.sub('[^ACGT]','Z',sequence)
    sequence_data_for_encode = np.array(list(sequence))
    sequence_data_for_encode=sequence_data_for_encode.reshape(-1,1)
    model = [['A'],['G'],['C'],['T'],['Z']]
    enc = OneHotEncoder()
    enc.fit(model)
    enc.transform(model).toarray()
    max_len = 300
    sequence_matrix = np.zeros((1,max_len*5))
    for i in range(len(sequence)):
        tempdata = enc.transform(sequence_data_for_encode).toarray()
        additional = np.zeros((1,max_len*5 - np.size(tempdata)))
        sequence_matrix = np.hstack((tempdata.flatten(),additional.flatten()))
    return torch.from_numpy(sequence_matrix.reshape(300,5)).to(torch.float32)

def inverton_search(tab_path, inverton_path,model_path,inverton_possibility_path):
    if os.path.getsize(tab_path)==0:
        print("#### The ir file is empty! ####") 
    tab = pd.read_table(tab_path)
    model=torch.load(model_path)      
    inverton = pd.DataFrame()
    posibility=[]
    id=[]
    for i in range(len(tab)):
        sequence=tab.iloc[i,6]+tab.iloc[i,7]+tab.iloc[i,8]
        
        if len(sequence) <= 300:
            b=one_hot(sequence)
            b=b.view(-1,1500)
            output=model(b)
            output=nn.Softmax(dim=1)(output).detach()
            _, prediction = torch.max(output, 1)
            prediction = prediction.numpy()[0]
            a=list(np.array(output).flatten())
            posibility.append(a)
            id.append(tab.iloc[i,0])
            if prediction == 1 and math.log10(output[0,1].item()/output[0,0].item())>15:
                inverton = pd.concat([inverton,tab.iloc[[i]]])
    c=np.array(posibility)
    inverton.rename(columns={0:"ID",1:"Scaffold",2:"PosA",3:"PosB",4:"PosC",5:"PosD",6:"IrA",7:"Mid",8:"IrB"})
    inverton.to_csv(inverton_path, sep='\t', index=False)
    result=pd.DataFrame({'ID':id,"positive":c[:,1],"negative":c[:,0]})
    result_path=inverton_possibility_path
    result.to_csv(result_path,sep='\t', index=False)

def deepinverton_irfinder(args):
    reffile = args.reffile
    prefix = args.prefix
    if os.path.exists(args.result_dirpath) is False:
        os.mkdir(args.result_dirpath)
    irfile = os.path.join(args.result_dirpath,prefix+'_ir.txt')
    if args.gcrange is not None:
        mingc = args.gcrange[0]
        maxgc = args.gcrange[1]
    einvertedparam = args.einvertedparam
    homopolymer = args.homopolymer
    maxmis = args.maxmis
    maxIR = args.maxIR
    f = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
    tmpout = f.name
    f.close()
    if einvertedparam is None:
        # if the einverted parameter is unspecified
        print("#### Now is finding inverted repeat ####")
        cmd = '''
        einverted -maxrepeat 750  -gap 100 -threshold 51 -match 5 -mismatch -9 -outfile {out}.51.outfile -outseq {out}.51.outseq -sequence {ref}
        einverted -maxrepeat 750  -gap 100 -threshold 75 -match 5 -mismatch -15 -outfile {out}.75.outfile -outseq {out}.75.outseq -sequence {ref}
        
        awk 'BEGIN{{OFS="\\t";ORS="";pass=0}}{{
            if(NR%5==2){{
                split($4,a,"/");
                if(a[2] <= 45 && (a[1]==a[2]) || (a[1]+1==a[2] && a[2] >=13) || (a[1]+2==a[2] && a[2] >=19)){{pass=1}}else{{pass=0}}
                sub(":","",$1);
                if(pass){{print $1"\\t"}}
            }}else if(NR%5==3 && pass ){{print $1-1,$3"\\t"}} else if(NR%5==0 && pass ){{print $3-1,$1"\\n"}}
        }}' {out}.51.outfile | awk '$4-$3>30 ' >{out}.pos.51.tab
        
        awk 'BEGIN{{OFS="\\t";ORS="";pass=0}}{{
            if(NR%5==2){{
                split($4,a,"/");
                if(a[2] <= 45 && (a[1]==a[2]) || (a[1]+1==a[2] && a[2] >=13) || (a[1]+2==a[2] && a[2] >=19)){{pass=1}}else{{pass=0}}
                sub(":","",$1);
                if(pass){{print $1"\\t"}}
            }}else if(NR%5==3 && pass ){{print $1-1,$3"\\t"}} else if(NR%5==0 && pass ){{print $3-1,$1"\\n"}}
        }}' {out}.75.outfile | awk '$4-$3>30 ' >{out}.pos.75.tab
        
        
        awk 'BEGIN{{OFS="\\t"}}{{print $1,$2,$5,$0}}' {out}.pos.51.tab |sortBed  > {out}.a.bed
        awk 'BEGIN{{OFS="\\t"}}{{print $1,$2,$5,$0}}' {out}.pos.75.tab |sortBed  > {out}.b.bed
        intersectBed   -a {out}.a.bed  -b {out}.b.bed  -v|cat - {out}.b.bed|cut -f 4- > {out}.pos.tab
        
        rm -rf {out} {out}.a.bed {out}.b.bed {out}.pos.51.tab {out}.pos.75.tab {out}.51.outfile  {out}.51.outseq  {out}.75.outfile  {out}.75.outseq '''.format(
            out=tmpout, ref=reffile)
        os.system(cmd)

    else:
        cmd = '''
        einverted  {einvertedparam} -outfile {out}.outfile -outseq {out}.outseq -sequence {ref}
        awk 'BEGIN{{OFS="\\t";ORS="";pass=0}}{{
            if(NR%5==2){{
                split($4,a,"/");
                if(a[2] <= {maxIR} && (a[2] - a[1] <= {maxmis} )){{pass=1}}else{{pass=0}}
                sub(":","",$1);
                if(pass){{print $1"\\t"}}
            }}else if(NR%5==3 && pass ){{print $1-1,$3"\\t"}} else if(NR%5==0 && pass ){{print $3-1,$1"\\n"}}
        }}' {out}.outfile  >{out}.pos.tab
        
        rm -rf {out} {out}.outfile  {out}.outseq  '''.format(
            out=tmpout,
            ref=reffile,
            maxIR=maxIR,
            maxmis=maxmis,
            einvertedParam=einvertedparam)
        os.system(cmd)
            
    seq_dict = SeqIO.to_dict(SeqIO.parse(reffile, "fasta"))
    lines = [x.rstrip().split("\t") for x in open(tmpout + ".pos.tab")]
    outfile = open(irfile, 'w+')
    print("ID"+"\t"+"Scaffold"+"\t"+"PosA"+"\t"+"PosB"+"\t"+"PosC"+"\t"+"PosD"+"\t"+"IrA"+"\t"+"Mid"+"\t"+"IrB",file=outfile)
    for each_line in lines:
        accept = 1
        each_seq = seq_dict[each_line[0]]
        posA = int(each_line[1])
        posB = int(each_line[2])
        posC = int(each_line[3])
        posD = int(each_line[4])
        each_ID='-'.join(each_line)
        each_line.insert(0,each_ID)
        

        left_seq = each_seq[posA:posB]
        right_seq = each_seq[posC:posD]
        mid_seq = each_seq[posB:posC]

        Lgc = GC(left_seq.seq)
        Rgc = GC(right_seq.seq)

        if homopolymer and \
                len(re.findall(r'([ACGT])\1{4,}', str(left_seq.seq))) > 0 and \
                len(re.findall(r'([ACGT])\1{4,}', str(right_seq.seq))) > 0:
                # if homopolymer filter is specified
            accept = 0

        if args.gcrange is not None and \
                (Lgc < mingc or Rgc < mingc or Lgc > maxgc or Rgc > maxgc):
                # if GC ratio filter is specified
            accept = 0

        if accept:
            print ("\t".join(each_line)+"\t"+left_seq.seq + \
                "\t"+mid_seq.seq+"\t" + right_seq.seq ,file=outfile)
    os.remove(tmpout + ".pos.tab")
    
def deepinverton_invertonfider(prefix,result_dirpath,model_path):
    print("#### Now is dentifying invertons ####")
    prefix = prefix
    irfile = os.path.join(result_dirpath,prefix+'_ir.txt')
    inverton_path=os.path.join(result_dirpath,prefix+'_inverton.txt')
    inverton_possibility_path=os.path.join(result_dirpath,prefix+'_ir_possibility.txt')
    model_path=model_path
    inverton_search(irfile, inverton_path,model_path,inverton_possibility_path)

def is_tool(name):
    """Check whether `name` is on PATH."""
    from distutils.spawn import find_executable
    return find_executable(name) is not None


if __name__ == "__main__":
    parser = ArgumentParser(
        description='Identifiy invertible regions in genomic sequence')
    parser.add_argument(
        '-f',
        '--fasta',
        help='input genome sequence file in fasta format',
        required=True,
        dest='reffile',
        metavar='') 
    parser.add_argument(
        '-x',
        '--prefix',
        help='the prefix of filename for output table ',
        required=True,
        dest='prefix',
        metavar='')
    parser.add_argument(
        '-e',
        '--einv',
        help='einverted parameters, if unspecified run with PhaseFinder default pipeline',
        required=False,
        dest='einvertedparam',
        metavar='')
    parser.add_argument(
        '-m',
        '--mismatch',
        help='max number of mismatches allowed between IR pairs,used with -einv (default:3)',
        type=int,
        required=False,
        dest='maxmis',
        default=3,
        metavar='')
    parser.add_argument(
        '-r',
        '--IRsize',
        help='max size of the inverted repeats, used with -einv (default:50)',
        type=int,
        required=False,
        dest='maxIR',
        default=50,
        metavar='')
    parser.add_argument(
        '-g',
        '--gcrange',
        help='the minimum and maximum value of GC ratio',
        nargs=2,
        type=float,
        required=False,
        dest='gcrange',
        metavar='')
    parser.add_argument(
        '-p',
        '--polymer',
        help='Remove homopolymer inverted repeats',
        action="store_true",
        dest='homopolymer')
    parser.add_argument(
        '-d',
        '--model',
        help='the path of deepinverton model',
        default=r'/data4/wenjiejie/Graduation_Project/add_data/model/data/extend/remove_extend2/linear1/fold1/model/0.01/6574_vali_roc_0.9119400991165698.pth',
        required=False,
        dest='model_path')
    parser.add_argument(
        '-o',
        '--outdir',
        help='the dir path of inverton result',
        default='',
        required=True,
        dest='result_dirpath'
    )
    parser.set_defaults(func=deepinverton_irfinder)


    if not is_tool("einverted"):
        print("tool {i} is not installed".format(i="einverted"))
        sys.exit(0)

    args = parser.parse_args()

    if  args.gcrange is not None:
        if not len(args.gcrange) == 2:
            raise Exception(
                "Specifiy the minimal and maximal value of the the GC range")
        else:
            minValue = args.gcrange[0]
            maxValue = args.gcrange[1]
            if 0 >= minValue or minValue > maxValue or maxValue > 100:
                raise Exception(
                    "The range should be between 0 and 100 and minimal value should be bigger than maximal"
                )
    args.func(args)
    deepinverton_invertonfider(args.prefix,args.result_dirpath,args.model_path)

