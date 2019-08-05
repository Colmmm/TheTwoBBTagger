from root_pandas import read_root
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, classification_report

class root_df:
    def __init__(self, path, columns, index_type='twobody', specific_ids=None, flatten=None, chunksize=None):
        self = read_root(paths=path, columns=columns, flatten=flatten , chunksize=chunksize)
        if index_type == 'event':
            self.index = self.apply(lambda x:str(int(x.runNumber)) + str(int(x.eventNumber)), axis=1 )
        elif index_type == 'twobody':
            self.index = self.apply(lambda x:str(int(x.runNumber)) + str(int(x.eventNumber))+'-'+str(int(x.nCandidate)), axis=1 )
        elif index_type == 'extratracks' and flatten!=None:
            self.index = self.apply(lambda x: str(int(x.runNumber)) + str(int(x.eventNumber)) + '-' + str(int(x.nCandidate))+'-'+str(int(x.__array_index)), axis=1)
        elif index_type == 'extratracks' and flatten==None:
            raise ValueError('You cant have the "extratracks" index if you dont flatten out any columns!')
        else:
            raise ValueError('The index type has to be either per "event", "twobody" or by "extratracks"!!!')
        if specific_ids!=None:
            self = self.loc[specific_ids, :]


def CV(df, target, test_size=0.33, nfolds=5, random_seed=42):
    feats = [c for c in df.columns not in target]
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
        oof.iloc[train_idx] = model.predict_proba(X_train[train_idx])[:,1]
        preds += model.predict_proba((X_test))[:,1] / skf.n_splits

    print('Cross Validation complete...\n\n\n')
    print('TRAIN OOF EVALUATION:\n\n')
    print(classification_report(y_train, round(oof)))
    print('\n\n\nTEST PREDS SCORE:\n\n')
    print(classification_report(y_test, round(preds)))

    all_preds = pd.concat([oof, preds])

    return all_preds

def firstStage(path, path4feats, path4ids, threshold = 0.16):
    ids = pd.Series.from_csv(path4ids).to_list()
    feats4FirstStage = pd.Series.from_csv(path4feats).to_list()
    df = root_df(path=path, columns= ids + feats4FirstStage, flatten=None, chunksize=None, index_type='twobody')
    TB_probs = CV(df=df, target = 'TwoBody_FromSameB', test_size=0.33, nfolds=5, random_seed=42)
    promising_TBs = TB_probs[TB_probs>threshold]
    return promising_TBs

def secondStage(path, path4ids, path4feats_1, path4feats_2, threshold=0.5):
    ids = pd.Series.from_csv(path4ids).to_list()
    feats4FirstStage = pd.Series.from_csv(path4feats_1).to_list()
    feats4SecondStage = pd.Series.from_csv(path4feats_2).to_list()
    df = root_df(path=path, columns=ids+ feats4FirstStage + feats4SecondStage, flatten=feats4SecondStage, chunksize=None, index_type='extratracks')
    ET_probs =  CV(df=df, target = 'TwoBody_FromSameB', test_size=0.33, nfolds=5, random_seed=42)
    promising_ETs = ET_probs[ET_probs>threshold]
    return promising_ETs


