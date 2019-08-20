from root_pandas import read_root
import pandas as pd
from tqdm import tqdm
from lof import LOF
from dataobjects import twoBBdf
from names_dict import GRAPHS1_DICT
import gc ; gc.enable()
from matplotlib import pyplot as plt
import seaborn as sns

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
    df4analysis = LOF(graphDF, generator=True)
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 3, 1)
    plt.title('PION TRACKS')
    plt.xlim(0, 4000)
    data = df4analysis.query('Track1_TruePion==1')
    dist1 = data.query('Track1_Charge*SignalB_ID == 521.0').Etrack1
    dist2 = data.query('Track1_Charge*SignalB_ID == -521.0').Etrack1
    sns.distplot(dist1, color="skyblue", label='Track1_Charge*SignalB_ID = 521.0')
    sns.distplot(dist2, color="red", label='Track1_Charge*SignalB_ID = -521.0')
    plt.legend()
    print(data.shape)

    plt.subplot(2, 3, 2)
    plt.title('MUON TRACKS')
    plt.xlim(0, 4000)
    data = df4analysis.query('Track1_TrueMuon==1')
    dist1 = data.query('Track1_Charge*SignalB_ID == 521.0').Etrack1
    dist2 = data.query('Track1_Charge*SignalB_ID == -521.0').Etrack1
    sns.distplot(dist1, color="orange", label='Track1_Charge*SignalB_ID = 521.0')
    sns.distplot(dist2, color="green", label='Track1_Charge*SignalB_ID = -521.0')
    plt.legend()
    print(data.shape)

    plt.subplot(2, 3, 3)
    plt.title('ELECTRON TRACKS')
    plt.xlim(0, 4000)
    data = df4analysis.query('Track2_TrueElectron==1')
    dist1 = data.query('Track2_Charge*SignalB_ID == 521.0').Etrack1
    dist2 = data.query('Track2_Charge*SignalB_ID == -521.0').Etrack1
    sns.distplot(dist1, color="purple", label='Track2_Charge*SignalB_ID = 521.0')
    sns.distplot(dist2, color="yellow", label='Track2_Charge*SignalB_ID = -521.0')
    plt.legend()
    print(data.shape)

    plt.subplot(2, 3, 4)
    plt.title('KAON TRACKS')
    plt.xlim(0, 4000)
    data = df4analysis.query('Track2_TrueKaon==1')
    dist1 = data.query('Track2_Charge*SignalB_ID == 521.0').Etrack1
    dist2 = data.query('Track2_Charge*SignalB_ID == -521.0').Etrack1
    sns.distplot(dist1, color="black", label='Track1_Charge*SignalB_ID = 521.0', bins=50)
    sns.distplot(dist2, color="purple", label='Track1_Charge*SignalB_ID = -521.0', bins=120)
    plt.legend()
    print(data.shape)

    plt.subplot(2, 3, 5)
    plt.title('PROTON TRACKS')
    plt.xlim(0, 4000)
    data = df4analysis.query('Track2_TrueProton==1')
    dist1 = data.query('Track2_Charge*SignalB_ID == 521.0').Etrack1
    dist2 = data.query('Track2_Charge*SignalB_ID == -521.0').Etrack1
    sns.distplot(dist1, color="black", label='Track1_Charge*SignalB_ID = 521.0', bins=50)
    sns.distplot(dist2, color="purple", label='Track1_Charge*SignalB_ID = -521.0', bins=120)
    plt.legend()
    print(data.shape)

    plt.savefig('lof_plots.png')


if __name__ == '__main__':
    main()



