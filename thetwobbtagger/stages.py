from training import CV
import pandas as pd
from root_pandas import read_root
from sklearn.metrics import accuracy_score
from utils import score_combiner
train_path = '../TaggingJpsiK2012_tiny_fix_fix.root'
test_path = '../TaggingJpsiK2012_tiny_fix_fixSlice2.root'

def firstStage(train_TBs, test_TBs ,threshold, random_seed=42):
    print('\nFirst Stage starting...\n\n')
    train_probs, test_probs = CV(train_twoBBdf=train_TBs, test_twoBBdf=test_TBs, nfolds=2, random_seed=random_seed, array_index=False )
    train_probs = train_probs[train_probs>threshold] ; test_probs = test_probs[test_probs>threshold]

    TB_w_ETs = [r for r in train_probs.index if r not in ['531373215304339-1', '531373215305125-0', '531373215305125-1']]
    train_probs = train_probs.loc[TB_w_ETs]

    TB_w_ETs = [r for r in test_probs.index if r not in ['5756867770067-1']]
    test_probs = test_probs.loc[TB_w_ETs]

    print('\n\nFirst Stage Complete!!!\n\n')
    return train_probs, test_probs

def secondStage(train_ETs, test_ETs, threshold, random_seed=42):
    print('\nSecond Stage Starting...\n')
    train_probs, test_probs = CV(train_twoBBdf=train_ETs, test_twoBBdf=test_ETs, nfolds=2, random_seed=random_seed, array_index=True)
    train_probs = train_probs[train_probs>threshold] ; test_probs = test_probs[test_probs>threshold]
    print('\n\nSecond Stage Complete!!!\n\n')
    return train_probs, test_probs


def preprocess(TAG_df, TB_scores, path):
    ids = ['runNumber', 'eventNumber', 'nCandidate']
    target = ['SignalB_ID']
    TAG_labels = read_root(paths=path, columns=ids + target)
    TAG_labels.index = TAG_labels.apply(
        lambda x: str(int(x.runNumber)) + str(int(x.eventNumber)) + '-' + str(int(x.nCandidate)), axis=1)
    TAG_labels = TAG_labels.replace(521, 1);
    TAG_labels = TAG_labels.replace(-521, 0)
    TAG_labels = TAG_labels.loc[TB_scores.index]

    event_ids = TAG_labels.apply(lambda x: str(int(x.runNumber)) + str(int(x.eventNumber)), axis=1)

    TAG_df = pd.concat([TAG_df, TAG_labels], axis=1).loc[TB_scores.index]

    return TAG_df, event_ids, TAG_labels



def thirdStage(train_TAG_df, test_TAG_df,  train_TB_scores, test_TB_scores, train_path=train_path, test_path=test_path, random_seed=42):
    print('\n\nStarting Third Stage...\n\n')
    ids = ['runNumber', 'eventNumber', 'nCandidate']
    train_TAG_df, train_event_ids, train_TAG_labels = preprocess(TAG_df=train_TAG_df, TB_scores=train_TB_scores, path=train_path)
    test_TAG_df, test_event_ids, test_TAG_labels = preprocess(TAG_df=test_TAG_df, TB_scores=test_TB_scores, path=test_path)

    train_TAG_probs, test_TAG_probs = CV(train_twoBBdf=train_TAG_df.drop(columns=ids + ['TB_id'], axis=0), test_twoBBdf=test_TAG_df.drop(columns=ids + ['TB_id'], axis=0),nfolds=2, random_seed=random_seed, justdf=True)

    TAG_preds = pd.concat([test_TAG_probs, test_TB_scores, test_event_ids, test_TAG_labels.SignalB_ID], axis=1);
    TAG_preds.columns = ['TAG_scores', 'TB_scores', 'event_id', 'label']
    per_event_TAG = TAG_preds.groupby('event_id').apply(score_combiner)
    per_event_TAG = per_event_TAG.groupby('event_id').mean()

    per_event_TAG = per_event_TAG.drop(per_event_TAG.query('label!=0.0 and label!=1.0').index, axis=0)

    print('\n\nThird Stage Complete!!\n')
    acc = accuracy_score(per_event_TAG.label, round(per_event_TAG.pred)) ; print('\nThe final Tagging Accuracy:\n{0}'.format(acc))
    tag_eff = 0.89*(per_event_TAG.shape[0]/1114) ; print('\nTagging efficiency:\n{0}'.format(tag_eff))
    print('\nOVERALL TAGGING POWER:\n{0}'.format(  tag_eff*(1-2*(1-acc))**2)  )

    return per_event_TAG

