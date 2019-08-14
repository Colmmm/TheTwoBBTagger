from training import CV
import pandas as pd
from root_pandas import read_root
from sklearn.metrics import accuracy_score
from utils import score_combiner
path = '../TaggingJpsiK2012_tiny_fix_fix.root'

def firstStage(TBs, threshold, random_seed=42):
    print('\nFirst Stage starting...\n\n')
    probs = CV(twoBBdf=TBs, test_size=0.33, nfolds=8, random_seed=random_seed, array_index=False )
    promising_probs = probs[probs>threshold]
    print('\n\nFirst Stage Complete!!!\n\n')
    return promising_probs

def secondStage(ETs, threshold, random_seed=42):
    print('\nSecond Stage Starting...\n')
    probs = CV(twoBBdf=ETs, test_size=0.33, nfolds=8, random_seed=random_seed, array_index=True)
    promising_probs = probs[probs>threshold]
    print('\n\nSecond Stage Complete!!!\n\n')
    return promising_probs

def thirdStage(TAG_df, TB_scores, path=path, random_seed=42):
    print('\n\nStarting Third Stage...\n\n')
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

    TAG_probs = CV(twoBBdf=TAG_df.drop(columns=ids + ['TB_id'], axis=0), test_size=0.33, nfolds=8,
                   random_seed=random_seed, justdf=True)

    TAG_preds = pd.concat([TAG_probs, TB_scores, event_ids, TAG_labels.SignalB_ID], axis=1);
    TAG_preds.columns = ['TAG_scores', 'TB_scores', 'event_id', 'label']
    per_event_TAG = TAG_preds.groupby('event_id').apply(score_combiner)
    per_event_TAG = per_event_TAG.groupby('event_id').mean()

    per_event_TAG = per_event_TAG.drop(per_event_TAG.query('label!=0.0 and label!=1.0').index, axis=0)

    print(accuracy_score(per_event_TAG.label, round(per_event_TAG.pred)))
    return per_event_TAG

