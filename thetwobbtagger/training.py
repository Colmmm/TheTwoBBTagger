from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, classification_report
from ml_insights import prob_calibration_function
import lightgbm as lgb

def CV(train_twoBBdf, test_twoBBdf, nfolds=7, random_seed = 42, justdf=False):
    #retrieves the df for the MVA
    train_df_generator = train_twoBBdf.get_MVAdf_generator()
    test_df_generator = test_twoBBdf.get_MVAdf_generator()

    #as we are doing the training and predicti
    all_preds = pd.Series()
    all_labels = pd.Series()

    model = None
    lgb_params = {
        'keep_training_booster': True,
        'objective': 'binary',
        'verbose_eval': -1,
        'verbose': -1
    }

    for train_df, test_df in zip(train_df_generator, test_df_generator):
        if train_twoBBdf.label == ['TwoBody_Extra_FromSameB']:
            feats = [c for c in train_df.columns if c not in train_df.label + train_df.ids + ['__array_index']]
        if train_twoBBdf.label == ['TwoBody_FromSameB']:
            feats = [c for c in train_df.columns if c not in train_df.label + train_df.ids]
        X_train = train_df[feats]
        y_train = train_df[train_twoBBdf.label]
        X_test = test_df[feats]
        y_test = test_df[train_twoBBdf.label]
        ids1 = X_train.index ; ids2 = X_test.index #we need to remember the ids as we will be converting data from pandas df to np.arrays

        #initalise empty series where the predictions will be inputted
        preds = pd.Series(np.zeros(shape=y_test.shape[0]), index=ids2)
        labels = y_test

        #we join the data together again so we can normalise it, and then we split up again
        all_data = pd.concat([X_train, X_test])
        norm_data = StandardScaler().fit_transform(all_data)
        X_train = norm_data[:X_train.shape[0]] ; y_train = y_train.to_numpy()
        X_test = norm_data[X_train.shape[0]:] ; y_test = y_test.to_numpy()

        model = lgb.train(lgb_params,
                          # Pass partially trained model:
                          init_model=model,
                          train_set=lgb.Dataset(X_train, y_train),
                          valid_sets=lgb.Dataset(X_test, y_test))

        preds.loc[ids2] = model.predict(X_test)

        all_preds = pd.concat([all_preds, preds])
        all_labels = pd.concat([all_labels, labels])
        





    print(oof.shape, preds.shape)
    #calibrating the output of the ML algorithm
    print('\nCalibrating...\n')
    calib_function = prob_calibration_function(y_train, oof)
    oof_calib = pd.Series(calib_function(oof), index=ids1)
    preds_calib = pd.Series(calib_function(preds), index=ids2)
    print('\nCalibration Complete!\n')

    #Calculating the performance of the model
    print('\n\nCross Validation complete!!!\n\n')
    print('\nTRAIN OOF EVALUATION:\n')
    print('\nROC_AUC_SCORE:\n{0}\n\n\nPRECISION_SCORE:\n{1}\n\n\nRECALL_SCORE:\n{2}\n'.format(
        roc_auc_score(y_train, oof), precision_score(y_train, round(oof)), recall_score(y_train, round(oof))))
    print('\n\nTEST PREDS SCORE:\n')
    print('\nROC_AUC_SCORE:\n{0}\n\n\nPRECISION_SCORE:\n{1}\n\n\nRECALL_SCORE:\n{2}\n'.format(
        roc_auc_score(y_test, preds), precision_score(y_test, round(preds)) , recall_score(y_test, round(preds))))


    return oof_calib, preds_calib
