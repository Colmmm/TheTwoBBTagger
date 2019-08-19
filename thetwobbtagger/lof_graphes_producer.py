from root_pandas import read_root
import pandas as pd
from tqdm import tqdm
from lof import LOF
import gc ; gc.enable()

path = '../TaggingJpsiK2012_tiny_fix_fix.root'

ids = [
    'runNumber'
    ,'eventNumber'
    ,'nCandidate']

cols4lof = [
    "TwoBody_M",
    "TwoBody_PE",
    "TwoBody_PX",
    "TwoBody_PY",
    "TwoBody_PZ",
    "TwoBody_ENDVERTEX_X",
    "TwoBody_ENDVERTEX_Y",
    "TwoBody_ENDVERTEX_Z",
    "TwoBody_OWNPV_X",
    "TwoBody_OWNPV_Y",
    "TwoBody_OWNPV_Z",
    "Track1_PX",
    "Track1_PY",
    "Track1_PZ",
    "Track2_PX",
    "Track2_PY",
    "Track2_PZ",
    'Track1_ProbNNe',
    'Track1_ProbNNk',
    'Track1_ProbNNp',
    'Track1_ProbNNpi',
    'Track1_ProbNNmu',
    'Track1_ProbNNghost',
    'Track2_ProbNNe',
    'Track2_ProbNNk',
    'Track2_ProbNNp',
    'Track2_ProbNNpi',
    'Track2_ProbNNmu',
    'Track2_ProbNNghost']

cols4graphs = [
    'Track1_Charge',
    'Track2_Charge',
    'Track1_TrueKaon',
    'Track1_TrueMuon',
    'SignalB_ID',
    'Track2_TrueMuon' ,
    'Track2_TrueKaon',
    'TwoBody_FromSameB']


df_generator = read_root(paths=path,columns=cols4lof+cols4graphs, chunksize=250000)

whole_df = pd.DataFrame()
cols2keep = cols4graphs + ['Etrack1', 'Etrack2']

def chunk_processing(chunk_df):
    chunk_df.index = chunk_df.apply(lambda x: str(int(x.runNumber)) + str(int(x.eventNumber)) + '-' + str(int(x.nCandidate)), axis=1)
    chunk_df = chunk_df.query('TwoBody_FromSameB==1')
    chunk_df = LOF(chunk_df)
    chunk_df = chunk_df.loc[:, cols2keep]
    chunk_df['Track1_Charge*SignalB_ID'] = chunk_df.apply(lambda x: x.Track1_Charge * x.SignalB_ID, axis=1)
    chunk_df['Track2_Charge*SignalB_ID'] = chunk_df.apply(lambda x: x.Track2_Charge * x.SignalB_ID, axis=1)
    return chunk_df


for chunk_df in tqdm(df_generator, unit='chunks'):
    chunk_df = chunk_processing(chunk_df)
    whole_df = pd.concat([whole_df, chunk_df])
    del chunk_df
    gc.collect()
    print(whole_df.shape[0])

whole_df.to_csv('data4lofPLOTS.csv')



