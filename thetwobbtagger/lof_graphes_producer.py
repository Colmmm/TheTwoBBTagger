from root_pandas import read_root
import pandas as pd
from tqdm import tqdm
from lof import LOF
from dataobjects import twoBBdf
from names_dict import GRAPHS1_DICT
import gc ; gc.enable()
from matplotlib import pyplot as plt
import seaborn as sns
import time

path = '../TaggingJpsiK2012_tiny_fix_fix.root'


def main():
    start_time = time.time()
    graphDF = twoBBdf(path = path, dict = GRAPHS1_DICT, chunk_size=5000)
    df4analysis = LOF(graphDF, generator=True)
    df4analysis.to_csv('data4_lof_PLOTS.csv')
    df4analysis = df4analysis.query('Etrack1<8000')

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
    sns.distplot(dist1, color="black", label='Track1_Charge*SignalB_ID = 521.0')
    sns.distplot(dist2, color="purple", label='Track1_Charge*SignalB_ID = -521.0')
    plt.legend()
    print(data.shape)

    plt.subplot(2, 3, 5)
    plt.title('PROTON TRACKS')
    plt.xlim(0, 4000)
    data = df4analysis.query('Track2_TrueProton==1')
    dist1 = data.query('Track2_Charge*SignalB_ID == 521.0').Etrack1
    dist2 = data.query('Track2_Charge*SignalB_ID == -521.0').Etrack1
    sns.distplot(dist1, color="black", label='Track1_Charge*SignalB_ID = 521.0')
    sns.distplot(dist2, color="purple", label='Track1_Charge*SignalB_ID = -521.0')
    plt.legend()
    print(data.shape)

    plt.savefig('lof_plots.png')


if __name__ == '__main__':
    main()



