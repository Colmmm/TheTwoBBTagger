import pandas as pd
from root_pandas import read_root

class twoBBdf:
    def __init__(self, path, dict, categorical_feats=None, specific_TBs=pd.Series(), specific_ETs=pd.Series()):
        self.path =path
        self.ids = dict['ids']
        self.feats4MVA = dict['MVA_key']
        self.feats4LOF = dict['LOF_key']
        self.label = dict['label_key']
        self.flatfeats4MVA = dict['flatMVA_key']
        self.flatfeats4LOF = dict['flatLOF_key']
        self.specific_TBs = specific_TBs
        self.specific_ETs = specific_ETs
        if categorical_feats!=None:
            self.categorical_feats = categorical_feats
        if self.label == ['TwoBody_FromSameB']:
            self.index_function = lambda x: str(int(x.runNumber)) + str(int(x.eventNumber))+'-'+str(int(x.nCandidate))
            self.threemomentum4LOF = [["Track1_PX", "Track1_PY", "Track1_PZ"] , ["Track2_PX", "Track2_PY", "Track2_PZ"] ]
            self.probs4LOF = [['Track1_ProbNNp', 'Track1_ProbNNk', 'Track1_ProbNNpi', 'Track1_ProbNNe', 'Track1_ProbNNmu'],
                          ['Track1_ProbNNp', 'Track1_ProbNNk', 'Track1_ProbNNpi', 'Track1_ProbNNe', 'Track1_ProbNNmu']]
            self.tracknames4LOF = ['track1', 'track2']
        if self.label == ['TwoBody_Extra_FromSameB']:
            self.index_function = lambda x: str(int(x.runNumber)) + str(int(x.eventNumber))+'-'+str(int(x.nCandidate)) +'-'+str(int(x['__array_index']))
            self.threemomentum4LOF = [['TwoBody_Extra_Px', 'TwoBody_Extra_Py','TwoBody_Extra_Pz']]
            self.probs4LOF = [['TwoBody_Extra_NNp', 'TwoBody_Extra_NNk', 'TwoBody_Extra_NNpi', 'TwoBody_Extra_NNe', 'TwoBody_Extra_NNmu']]
            self.tracknames4LOF = ['extra_track']

    def get_MVAdf(self):
        MVAdf = read_root(paths=self.path, columns=self.ids+self.label+self.feats4MVA, flatten=self.flatfeats4MVA)
        MVAdf.index = MVAdf.apply(lambda x:str(int(x.runNumber)) + str(int(x.eventNumber))+'-'+str(int(x.nCandidate)), axis=1 )
        if self.specific_TBs.shape[0] != 0:
            MVAdf = MVAdf.loc[self.specific_TBs, :]
        MVAdf.index = MVAdf.apply(self.index_function, axis=1)
        if self.specific_ETs.shape[0]!=0:
            MVAdf = MVAdf.loc[self.specific_ETs, :]
        return MVAdf

    def get_LOFdf(self):
        LOFdf = read_root(paths=self.path, columns=self.ids + self.feats4LOF, flatten=self.flatfeats4LOF)
        LOFdf.index = LOFdf.apply(lambda x: str(int(x.runNumber)) + str(int(x.eventNumber)) + '-' + str(int(x.nCandidate)), axis=1)
        if self.specific_TBs.shape[0]!=0:
            LOFdf = LOFdf.loc[self.specific_TBs,:]
        LOFdf.index = LOFdf.apply(self.index_function, axis=1)
        if self.specific_ETs.shape[0]!=0:
            LOFdf = LOFdf.loc[self.specific_ETs, :]

        return LOFdf


