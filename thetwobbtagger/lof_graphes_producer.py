from root_pandas import read_root
import pandas as pd
from tqdm import tqdm
from lof import LOF
from dataobjects import twoBBdf
from names_dict import GRAPHS1_DICT
import gc ; gc.enable()

path = '../TaggingJpsiK2012_tiny_fix_fix.root'


cols2keep = ['SignalB_ID' , 'Track1_TrueMuon', 'Track2_TrueKaon', 'Track1_TrueKaon','Track2_TrueMuon',
             'Etrack1', 'Etrack2', 'Track1_Charge', 'Track2_Charge' ]

def chunk_processing(chunk_df):
    chunk_df.index = chunk_df.apply(lambda x: str(int(x.runNumber)) + str(int(x.eventNumber)) + '-' + str(int(x.nCandidate)), axis=1)
    chunk_df = chunk_df.query('TwoBody_FromSameB==1')
    chunk_df = LOF(chunk_df)
    chunk_df = chunk_df.loc[:, cols2keep]
    chunk_df['Track1_Charge*SignalB_ID'] = chunk_df.apply(lambda x: x.Track1_Charge * x.SignalB_ID, axis=1)
    chunk_df['Track2_Charge*SignalB_ID'] = chunk_df.apply(lambda x: x.Track2_Charge * x.SignalB_ID, axis=1)
    return chunk_df


class graph_df():

    def __init__(self, path, dict, chunk_size=25000):
        self = twoBBdf(path = path, dict = dict)
        self.generator = read_root(paths=self.path, columns=self.ids+self.feats4LOF+self.label, chunksize=chunk_size)


def main():
    graphDF = twoBBdf(path = path, dict = GRAPHS1_DICT, chunk_size=5000)
    LOF(graphDF, generator=True)


if __name__ == '__main__':
    main()



