import pandas as pd
from root_pandas import read_root

class twoBBdf:
    def __init__(self, path, dict, specific_TBs=pd.Series(), specific_ETs=pd.Series()):
        #the attributes below refer to the different kind of column/branch names stored in the name dictionary
        self.path =path
        self.ids = dict['ids']
        self.feats4MVA = dict['MVA_key']
        self.feats4LOF = dict['LOF_key']
        self.label = dict['label_key']
        self.flatfeats4MVA = dict['flatMVA_key']
        self.flatfeats4LOF = dict['flatLOF_key']
        #the specific_XXs attributes is for when you only want to read in specific TBs or ETs, filtered by their ids which is inputed as a panda series
        self.specific_TBs = specific_TBs
        self.specific_ETs = specific_ETs
        if self.label == ['TwoBody_FromSameB']:
            #this index_function is used to create the index, ie TB index for TBs and ET index for ETs
            self.index_function = lambda x: str(int(x.runNumber)) + str(int(x.eventNumber))+'-'+str(int(x.nCandidate))
            # depending if its TB or ET, we need to define the below attributes differently, which is used in the lof calculation, just as the names are slighty different
            self.threemomentum4LOF = [["Track1_PX", "Track1_PY", "Track1_PZ"] , ["Track2_PX", "Track2_PY", "Track2_PZ"] ]
            self.probs4LOF = [['Track1_ProbNNp', 'Track1_ProbNNk', 'Track1_ProbNNpi', 'Track1_ProbNNe', 'Track1_ProbNNmu'],
                          ['Track1_ProbNNp', 'Track1_ProbNNk', 'Track1_ProbNNpi', 'Track1_ProbNNe', 'Track1_ProbNNmu']]
            self.tracknames4LOF = ['track1', 'track2']
        if self.label == ['TwoBody_Extra_FromSameB']:
            self.index_function = lambda x: str(int(x.runNumber)) + str(int(x.eventNumber))+'-'+str(int(x.nCandidate)) +'-'+str(int(x['__array_index']))
            self.threemomentum4LOF = [['TwoBody_Extra_Px', 'TwoBody_Extra_Py','TwoBody_Extra_Pz']]
            self.probs4LOF = [['TwoBody_Extra_NNp', 'TwoBody_Extra_NNk', 'TwoBody_Extra_NNpi', 'TwoBody_Extra_NNe', 'TwoBody_Extra_NNmu']]
            self.tracknames4LOF = ['extra_track']

    def get_MVAdf_generator(self, chunk_size=10000):
        '''this method creates the df with the right columns/branches from the root df for the MVA procedure'''
        MVAdf_generator = read_root(paths=self.path, columns=self.ids + self.feats4MVA + self.label, flatten=self.flatfeats4MVA, chunksize=chunk_size)

        for chunkdf in MVAdf_generator:
            #always change index to be a TB index first, just in case we only want a select few of TBs, which we cant do if we have a ET index
            chunkdf.index = chunkdf.apply(lambda x:str(int(x.runNumber)) + str(int(x.eventNumber))+'-'+str(int(x.nCandidate)), axis=1 )
            #if specific_TBs is not empty then we need to filter out the unwanted TBs by their id
            if self.specific_TBs.shape[0] != 0:
                chunkdf = chunkdf.loc[self.specific_TBs, :]
            #we then change the index according to whether were dealing with TBs or ETs, if its TBs then the index is essentially left unchanged
            chunkdf.index = chunkdf.apply(self.index_function, axis=1)
            #if specific_ETs is not empty that means we need to filter out and keep only the ETs asked for by using their ids
            if self.specific_ETs.shape[0]!=0:
                chunkdf = chunkdf.loc[self.specific_ETs, :]
            # below if statement is if were dealing with ETs and have an extra column '__array_index' added, which we need to remove for training purposes
            if self.label == ['TwoBody_Extra_FromSameB']:
                feats = [c for c in chunkdf.columns if c not in chunkdf.label+chunkdf.ids+['__array_index']]
            if self.label == ['TwoBody_FromSameB']:
                feats = [c for c in chunkdf.columns if c not in chunkdf.label+chunkdf.ids]
            X = chunkdf[feats]
            y = chunkdf[self.label]
            yield X, y

    def get_LOFdf_generator(self, chunk_size=10000):
        '''this method creates the df with the right columns/branches in which to perform the LOF calculation on, and require the COM variables for the TBs and ETs'''
        LOFdf_generator = read_root(paths=self.path, columns=self.ids + self.feats4LOF, flatten=self.flatfeats4LOF, chunksize=chunk_size)

        for chunkdf in LOFdf_generator:
            #always change index to be a TB index first, just in case we only want a select few of TBs, which we cant do if we have a ET index
            chunkdf.index = chunkdf.apply(lambda x: str(int(x.runNumber)) + str(int(x.eventNumber)) + '-' + str(int(x.nCandidate)), axis=1)
            # if specific_TBs is not empty then we need to filter out the unwanted TBs by their id
            if self.specific_TBs.shape[0]!=0:
                chunkdf = chunkdf.loc[self.specific_TBs,:]
            # we then change the index according to whether were dealing with TBs or ETs, if its TBs then the index is essentially left unchanged
            chunkdf.index = chunkdf.apply(self.index_function, axis=1)
            #if specific_ETs is not empty that means we need to filter out and keep only the ETs asked for by using their ids
            if self.specific_ETs.shape[0]!=0:
               chunkdf = chunkdf.loc[self.specific_ETs, :]

            yield chunkdf


