from root_pandas import read_root
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, classification_report
from names_dict import TB_dict, ET_dict


class twoBBdf:
    def __init__(self, path, dict, categorical_feats=None, specific_TBs=pd.Series()):
        self.path =path
        self.ids = dict['ids']
        self.feats4MVA = dict['MVA_key']
        self.feats4LOF = dict['LOF_key']
        self.label = dict['label_key']
        self.flatfeats4MVA = dict['flatMVA_key']
        self.flatfeats4LOF = dict['flatLOF_key']
        self.specific_TBs = specific_TBs
        if categorical_feats!=None:
            self.categorical_feats = categorical_feats
        if self.label == ['TwoBody_FromSameB']:
            self.index_function = lambda x: str(int(x.runNumber)) + str(int(x.eventNumber))+'-'+str(int(x.nCandidate))
        if self.label == ['TwoBody_Extra_FromSameB']:
            self.index_function = lambda x: str(int(x.runNumber)) + str(int(x.eventNumber))+'-'+str(int(x.nCandidate)) +'-'+str(int(x['__array_index']))
    def get_MVAdf(self):
        MVAdf = read_root(paths=self.path, columns=self.ids+self.label+self.feats4MVA, flatten=self.flatfeats4MVA)
        MVAdf.index = MVAdf.apply(lambda x:str(int(x.runNumber)) + str(int(x.eventNumber))+'-'+str(int(x.nCandidate)), axis=1 )
        if self.specific_TBs.shape[0] != 0:
            MVAdf = MVAdf.loc[self.specific_TBs, :]
        MVAdf.index = MVAdf.apply(self.index_function, axis=1)
        if self.label == ['TwoBody_Extra_FromSameB']:
            self.ids = self.ids + ['__array_index']
        return MVAdf

    def get_LOFdf(self):
        LOFdf = read_root(paths=self.path, columns=self.ids + self.feats4LOF, flatten=self.flatfeats4LOF)
        LOFdf.index = LOFdf.apply(lambda x: str(int(x.runNumber)) + str(int(x.eventNumber)) + '-' + str(int(x.nCandidate)), axis=1)
        if self.specific_TBs.shape[0]!=0:
            LOFdf = LOFdf.loc[self.specific_TBs,:]
        LOFdf.index = LOFdf.apply(self.index_function, axis=1)
        if self.label == ['TwoBody_Extra_FromSameB']:
            self.ids = self.ids + ['__array_index']
        return LOFdf

def CV(twoBBdf, test_size=0.33, nfolds=5, random_seed=42):
    df = twoBBdf.get_MVAdf()
    feats = [c for c in df.columns if c not in twoBBdf.label+twoBBdf.ids]
    print(feats)
    target = twoBBdf.label[0]
    X = df[feats]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    ids1 = X_train.index ; ids2 = X_test.index

    oof = pd.Series(np.zeros(shape=y_train.shape[0]), index=ids1)
    preds = pd.Series(np.zeros(shape=y_test.shape[0]), index=ids2)

    all_data = pd.concat([X_train, X_test])
    norm_data = StandardScaler().fit_transform(all_data)
    X_train = norm_data[:X_train.shape[0]] ; y_train = y_train.to_numpy()
    X_test = norm_data[X_train.shape[0]:] ; y_test = y_test.to_numpy()

    skf = KFold(n_splits=nfolds, random_state=random_seed)

    for train_idx, cv_index in tqdm(skf.split(X_train, y_train), total= skf.n_splits):
        model = LGBMClassifier()
        model.fit(X_train[train_idx], y_train[train_idx])
        oof.iloc[cv_index] = model.predict_proba(X_train[cv_index])[:,1]
        preds += model.predict_proba((X_test))[:,1] / skf.n_splits

    print('\n\nCross Validation complete...\n\n\n')
    print('TRAIN OOF EVALUATION:\n\n')
    print(classification_report(y_train, round(oof)))
    print('\n\n\nTEST PREDS SCORE:\n\n')
    print(classification_report(y_test, round(preds)))
    print(roc_auc_score(y_train, oof), precision_score(y_train, round(oof)), recall_score(y_train, round(oof)))
    print(roc_auc_score(y_test, preds), precision_score(y_test, round(preds)) , recall_score(y_test, round(preds)))

    all_preds = pd.concat([oof, preds])

    return all_preds


def firstStage(TBs, threshold, random_seed=42 ):
    print('\nFirst Stage starting...\n\n')
    probs = CV(twoBBdf=TBs, test_size=0.33, nfolds=8, random_seed=random_seed)
    promising_probs = probs[probs>threshold]
    print('\n\nFirst Stage Complete!!!\n\n')
    return promising_probs

def secondStage(ETs, threshold, random_seed=42):
    print('\nSecond Stage Starting...\n')
    probs = CV(twoBBdf=ETs, test_size=0.33, nfolds=8, random_seed=random_seed)
    promising_probs = probs[probs>threshold]
    print('\n\nSecond Stage Complete!!!\n\n')
    return promising_probs

def main():
    path = 'TaggingJpsiK2012_tiny_fix_fix.root'
    TBs = twoBBdf(path=path, dict=TB_dict)
    promisingTBs = firstStage(TBs, threshold=0.16, random_seed=42)
    ETs = twoBBdf(path=path, dict=ET_dict, specific_TBs=promisingTBs.index )
    promisingETs = secondStage(ETs, threshold=0.5, random_seed=42)



if __name__ == "__main__":
    main()