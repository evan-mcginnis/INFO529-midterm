import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser("CSV2NPZ converter")

parser.add_argument('-i', '--input', action="store", required=True, help="Input CSV")
parser.add_argument('-o', '--output', action="store", required=True, help="Output NPZ")

arguments = parser.parse_args()

data_csv=pd.read_csv(arguments.input,delimiter=',')

data_npz=np.array(data_csv)

np.savez_compressed(arguments.output,data=data_npz)



