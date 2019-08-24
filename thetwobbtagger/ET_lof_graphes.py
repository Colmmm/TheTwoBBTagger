from root_pandas import read_root
import pandas as pd
from tqdm import tqdm
from lof import LOF
from dataobjects import twoBBdf
from names_dict import GRAPHS2_DICT
import gc ; gc.enable()
from matplotlib import pyplot as plt
import seaborn as sns
import time

path = '../TaggingJpsiK2012_tiny_fix_fix.root'


def main():
    start_time = time.time()
    graphDF = twoBBdf(path = path, dict = GRAPHS2_DICT, chunk_size=5000)
    df4analysis = LOF(graphDF, generator=True)
    df4analysis.to_csv('ETs_PLOT4lof_data_onSAT.csv')
    df4analysis = df4analysis.query('Eextra_track<8000')

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 3, 1)
    plt.title('PION TRACKS')
    plt.xlim(0, 4000)
    data = df4analysis.query('TwoBody_Extra_TRUEPID==211')
    dist1 = data.query('TwoBody_Extra_CHARGE*SignalB_ID == 521.0').Eextra_track
    dist2 = data.query('TwoBody_Extra_CHARGE*SignalB_ID == -521.0').Eextra_track
    sns.distplot(dist1, color="skyblue", label='TwoBody_Extra_CHARGE*SignalB_ID = 521.0')
    sns.distplot(dist2, color="red", label='TwoBody_Extra_CHARGE*SignalB_ID = -521.0')
    plt.legend()
    print(data.shape)

    plt.subplot(2, 3, 2)
    plt.title('MUON TRACKS')
    plt.xlim(0, 4000)
    data = df4analysis.query('TwoBody_Extra_TRUEPID==13')
    dist1 = data.query('TwoBody_Extra_CHARGE*SignalB_ID == 521.0').Eextra_track
    dist2 = data.query('TwoBody_Extra_CHARGE*SignalB_ID == -521.0').Eextra_track
    sns.distplot(dist1, color="orange", label='TwoBody_Extra_CHARGE*SignalB_ID = 521.0')
    sns.distplot(dist2, color="green", label='TwoBody_Extra_CHARGE*SignalB_ID = -521.0')
    plt.legend()
    print(data.shape)

    plt.subplot(2, 3, 3)
    plt.title('ELECTRON TRACKS')
    plt.xlim(0, 4000)
    data = df4analysis.query('TwoBody_Extra_TRUEPID==11')
    dist1 = data.query('TwoBody_Extra_CHARGE*SignalB_ID == 521.0').Eextra_track
    dist2 = data.query('TwoBody_Extra_CHARGE*SignalB_ID == -521.0').Eextra_track
    sns.distplot(dist1, color="purple", label='TwoBody_Extra_CHARGE*SignalB_ID = 521.0')
    sns.distplot(dist2, color="yellow", label='TwoBody_Extra_CHARGE*SignalB_ID = -521.0')
    plt.legend()
    print(data.shape)

    plt.subplot(2, 3, 4)
    plt.title('KAON TRACKS')
    plt.xlim(0, 4000)
    data = df4analysis.query('TwoBody_Extra_TRUEPID==321')
    dist1 = data.query('TwoBody_Extra_CHARGE*SignalB_ID == 521.0').Eextra_track
    dist2 = data.query('TwoBody_Extra_CHARGE*SignalB_ID == -521.0').Eextra_track
    sns.distplot(dist1, color="black", label='TwoBody_Extra_CHARGE*SignalB_ID = 521.0')
    sns.distplot(dist2, color="purple", label='TwoBody_Extra_CHARGE*SignalB_ID = -521.0')
    plt.legend()
    print(data.shape)

    plt.subplot(2, 3, 5)
    plt.title('PROTON TRACKS')
    plt.xlim(0, 4000)
    data = df4analysis.query('TwoBody_Extra_TRUEPID==2212')
    dist1 = data.query('TwoBody_Extra_CHARGE*SignalB_ID == 521.0').Eextra_track
    dist2 = data.query('TwoBody_Extra_CHARGE*SignalB_ID == -521.0').Eextra_track
    sns.distplot(dist1, color="black", label='TwoBody_Extra_CHARGE*SignalB_ID = 521.0')
    sns.distplot(dist2, color="yellow", label='TwoBody_Extra_CHARGE*SignalB_ID = -521.0')
    plt.legend()
    print(data.shape)

    plt.savefig('plots_4_lof_ETs.png')


if __name__ == '__main__':
    main()


