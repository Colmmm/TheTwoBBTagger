from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, classification_report
from ml_insights import prob_calibration_function


def CV(twoBBdf, test_size=0.33, nfolds=5, random_seed=42, array_index=False, justdf=False):
    if justdf==True:
        df = twoBBdf
        target = 'SignalB_ID'
        feats = [c for c in df.columns if c not in target]
    else:
        df = twoBBdf.get_MVAdf()
        if array_index==True:
            feats = [c for c in df.columns if c not in twoBBdf.label+twoBBdf.ids+ ['__array_index']]
        feats = [c for c in df.columns if c not in twoBBdf.label+twoBBdf.ids]
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

    print('\nCalibrating...\n')
    calib_function = prob_calibration_function(y_train, oof)
    oof_calib = pd.Series(calib_function(oof), index=ids1)
    preds_calib = pd.Series(calib_function(preds), index=ids2)
    print('\nCalibration Complete!\n')

    print('\n\nCross Validation complete!!!\n\n')
    print('\nTRAIN OOF EVALUATION:\n')
    print('\nROC_AUC_SCORE:\n{0}\n\n\nPRECISION_SCORE:\n{1}\n\n\nRECALL_SCORE:\n{2}\n'.format(
        roc_auc_score(y_train, oof), precision_score(y_train, round(oof)), recall_score(y_train, round(oof))))
    print('\n\nTEST PREDS SCORE:\n')
    print('\nROC_AUC_SCORE:\n{0}\n\n\nPRECISION_SCORE:\n{1}\n\n\nRECALL_SCORE:\n{2}\n'.format(
        roc_auc_score(y_test, preds), precision_score(y_test, round(preds)) , recall_score(y_test, round(preds))))

    all_calib_preds = pd.concat([oof_calib, preds_calib])

    return all_calib_preds