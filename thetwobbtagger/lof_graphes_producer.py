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


class graph_df(twoBBdf):

    def generateData4graphes(self, chunk_size=250000):
        df_generator = read_root(paths=path, columns=self.ids+self.feats4LOF, chunksize=chunk_size)

        whole_df = pd.DataFrame()

        for chunk_df in tqdm(df_generator, unit='chunks'):
            chunk_df = chunk_processing(chunk_df)
            whole_df = pd.concat([whole_df, chunk_df])
            del chunk_df
            gc.collect()
            print(whole_df.shape[0])

        whole_df.to_csv('data4lofPLOTS.csv')

        return whole_df


def main():
    graphDF = graph_df(path = path, dict = GRAPHS1_DICT)
    graphDF.generateData4graphes()


if __name__ == '__main__':
    main()



