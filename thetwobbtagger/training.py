from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, classification_report
from ml_insights import prob_calibration_function

def CV(train_twoBBdf, test_twoBBdf, chunk_size, nfolds=5, random_seed = 42, array_index=False, justdf=False):
    #this if statement is for thirdStage, where we dont have a twoBBdf object for data but just a df, so needs to be treated differently
    if justdf==True:
        train_df = train_twoBBdf
        test_df = test_twoBBdf
        target = 'SignalB_ID'
        feats = [c for c in train_df.columns if c not in target]
    else:
        #retrieves the df for the MVA
        train_df = train_twoBBdf.get_MVAdf(chunk_size=chunk_size)
        test_df = test_twoBBdf.get_MVAdf(chunk_size=chunk_size)
        #below if statement is if were dealing with ETs and have an extra column '__array_index' added, which we need to remove for training purposes
        if array_index==True:
            feats = [c for c in train_df.columns if c not in train_twoBBdf.label+train_twoBBdf.ids+ ['__array_index']]
        #we need to remove the id like columns/branches for training, as well as the label
        feats = [c for c in train_df.columns if c not in train_twoBBdf.label+train_twoBBdf.ids]
        target = train_twoBBdf.label[0] #we have [0] as label is inputted as a list

    X_train = train_df[feats] ; y_train = train_df[target]
    X_test = test_df[feats] ; y_test = test_df[target]
    ids1 = X_train.index ; ids2 = X_test.index #we need to remember the ids as we will be converting data from pandas df to np.arrays

    #initalise empty series where the predictions will be inputted
    oof = pd.Series(np.zeros(shape=y_train.shape[0]), index=ids1)
    preds = pd.Series(np.zeros(shape=y_test.shape[0]), index=ids2)

    #we join the data together again so we can normalise it, and then we split up again
    all_data = pd.concat([X_train, X_test])
    norm_data = StandardScaler().fit_transform(all_data)
    X_train = norm_data[:X_train.shape[0]] ; y_train = y_train.to_numpy()
    X_test = norm_data[X_train.shape[0]:] ; y_test = y_test.to_numpy()

    #KFOLD
    skf = KFold(n_splits=nfolds, random_state=random_seed)
    #so we split data into nfolds and we train n different models that train and predict on different subections of training data, they all pred on test data though
    for train_idx, cv_index in tqdm(skf.split(X_train, y_train), total= skf.n_splits):
        model = LGBMClassifier()
        model.fit(X_train[train_idx], y_train[train_idx])
        oof.iloc[cv_index] = model.predict_proba(X_train[cv_index])[:,1]
        preds += model.predict_proba((X_test))[:,1] / skf.n_splits

    print(oof.shape, preds.shape)

    #Calculating the performance of the model
    print('\n\nCross Validation complete!!!\n\n')
    print('\nTRAIN OOF EVALUATION:\n')
    print('\nROC_AUC_SCORE:\n{0}\n\n\nPRECISION_SCORE:\n{1}\n\n\nRECALL_SCORE:\n{2}\n'.format(
        roc_auc_score(y_train, oof), precision_score(y_train, round(oof)), recall_score(y_train, round(oof))))
    print('\n\nTEST PREDS SCORE:\n')
    print('\nROC_AUC_SCORE:\n{0}\n\n\nPRECISION_SCORE:\n{1}\n\n\nRECALL_SCORE:\n{2}\n'.format(
        roc_auc_score(y_test, preds), precision_score(y_test, round(preds)) , recall_score(y_test, round(preds))))


    return oof, preds
