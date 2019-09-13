from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, classification_report
from ml_insights import prob_calibration_function
import lightgbm as lgb
import gc ; gc.enable()


def CV(train_twoBBdf, test_twoBBdf, chunk_size=1000,  nfolds=7, random_seed=42, justdf=False):
    # retrieves the df for the MVA
    train_df_generator = train_twoBBdf.get_MVAdf_generator(chunk_size=chunk_size)
    test_df_generator = test_twoBBdf.get_MVAdf_generator(chunk_size=chunk_size)

    # as we are doing the training and predictions in batches, we need to define a main preds list which then gets added to after each batch
    TRAIN_PREDS = pd.Series()
    TRAIN_LABELS = pd.Series()
    TEST_PREDS = pd.Series()
    TEST_LABELS = pd.Series()

    model = None
    lgb_params = {
        'keep_training_booster': False,
        'objective': 'binary',
        'verbose_eval': -1,
        'verbose': -1,
        'metric':'roc'
    }

    for train_df, test_df in tqdm(zip(train_df_generator, test_df_generator), unit='chunks'):
        # below if statement is if were dealing with ETs and have an extra column '__array_index' added, which we need to remove for training purposes
        if train_twoBBdf.label == ['TwoBody_Extra_FromSameB']:
            feats = [c for c in train_df.columns if c not in train_twoBBdf.label + train_twoBBdf.ids + ['__array_index']]
        if train_twoBBdf.label == ['TwoBody_FromSameB']:
            feats = [c for c in train_df.columns if c not in train_twoBBdf.label + train_twoBBdf.ids]
        X_train = train_df[feats]
        y_train = train_df[train_twoBBdf.label[0]]
        X_test = test_df[feats]
        y_test = test_df[train_twoBBdf.label[0]]
        ids1 = X_train.index; ids2 = X_test.index  # we need to remember the ids as we will be converting data from pandas df to np.arrays

        print('feats:{0}'.format(feats))

        # initalise empty series where the predictions will be inputted
        train_preds = pd.Series(np.zeros(y_train.shape[0]), index=ids1)
        train_labels = y_train
        test_preds = pd.Series(np.zeros(y_test.shape[0]), index=ids2)
        test_labels = y_test


        # we join the data together again so we can normalise it, and then we split up again
        all_data = pd.concat([X_train, X_test])
        scaler = StandardScaler().fit(X_train); norm_data = scaler.transform(all_data)
        X_train = norm_data[:X_train.shape[0]]; y_train = y_train.to_numpy().ravel()
        X_test = norm_data[X_train.shape[0]:]; y_test = y_test.to_numpy().ravel()

        model = lgb.train(lgb_params,
                          # Pass partially trained model:
                          init_model=model,
                          train_set=lgb.Dataset(X_train, y_train),
                          valid_sets=lgb.Dataset(X_test, y_test))

        train_preds.loc[ids1] = model.predict(X_train)
        test_preds.loc[ids2] = model.predict(X_test)

        TRAIN_PREDS = pd.concat([TRAIN_PREDS, train_preds])
        TRAIN_LABELS = pd.concat([TRAIN_LABELS, train_labels])
        TEST_PREDS = pd.concat([TEST_PREDS, test_preds])
        TEST_LABELS = pd.concat([TEST_LABELS, test_labels])

        del X_train, X_test, y_train, y_test, train_labels, test_labels
        gc.collect()

    # Calculating the performance of the model
    print('\n\nTraining complete!!!\n\n')
    print('\nTRAIN OOF EVALUATION:\n')
    print('\nROC_AUC_SCORE:\n{0}\n\n\nPRECISION_SCORE:\n{1}\n\n\nRECALL_SCORE:\n{2}\n'.format(
        roc_auc_score(TRAIN_LABELS, TRAIN_PREDS), precision_score(TRAIN_LABELS, round(TRAIN_PREDS)), recall_score(TRAIN_LABELS, round(TRAIN_PREDS))))
    print('\n\nTEST PREDS SCORE:\n')
    print('\nROC_AUC_SCORE:\n{0}\n\n\nPRECISION_SCORE:\n{1}\n\n\nRECALL_SCORE:\n{2}\n'.format(
        roc_auc_score(TRAIN_LABELS, TRAIN_PREDS), precision_score(TRAIN_LABELS, round(TRAIN_PREDS)), recall_score(TRAIN_LABELS, round(TRAIN_PREDS))))

    return TRAIN_PREDS, TEST_PREDS