import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#write a .csv file :
"""
columns:        word|segmentation|prdiction|metrics

run metrics for every word and write them in a .csv file and analise results
"""

def main(args):
    #to be continued


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Word by word metrics')
    parser.add_argument("--gold", help="Gold standard", required=True, type=str)
    parser.add_argument("--guess", help="Model output", required=True, type=str)
    parser.add_argument("--out", help="Outout file name", required=True, type=str)
    opt = parser.parse_args()
    main(opt)