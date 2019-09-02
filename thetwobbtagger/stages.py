from training import CV
import pandas as pd
from sklearn.metrics import accuracy_score
from utils import score_combiner, preprocess4TAGGING
train_path = '../TaggingJpsiK2012_tiny_fix_fix.root'
test_path = '../TaggingJpsiK2012_tiny_fix_fixSlice2.root'

def firstStage(train_TBs, test_TBs ,threshold, random_seed=42):
    """This function takes a TB's twobbdf object as its input, and then calculates the probabilities of how likely each
       TB comes from the decay of the Tagging B. These TBs are then filtered by getting rid of any TBs which have
       probabilities below the threshold (which is specified as an input parameter). This function then outputs a pandas
       series with a TBs index and the probabilities outputted by the TB MVA.
    """
    print('\nFirst Stage starting...\n\n')
    train_probs, test_probs = CV(train_twoBBdf=train_TBs, test_twoBBdf=test_TBs, nfolds=5, random_seed=random_seed, array_index=False )
    train_probs = train_probs[train_probs>threshold] ; test_probs = test_probs[test_probs>threshold]

    #at the moment, second stage cant deal with TBs without ETs, so I just get rid of the TBs without ETs for now
    TB_w_ETs = [r for r in train_probs.index if r not in ['531373215304339-1', '531373215305125-0', '531373215305125-1']]
    train_probs = train_probs.loc[TB_w_ETs]

    TB_w_ETs = [r for r in test_probs.index if r not in ['5756867770067-1']]
    test_probs = test_probs.loc[TB_w_ETs]

    print('\n\nFirst Stage Complete!!!\n\n')
    return train_probs, test_probs

def secondStage(train_ETs, test_ETs, threshold, random_seed=42):
    """This function takes an ET twobbdf object as its input, and then calculates the probabilities of whether or not
       each ET comes from the decay of the Tagging B. Then similar to the TB case, the ETs are filtered by getting rid
       of any ET which has a probability below the threshold (which is specified as an input parameter). This function
       then outputs
     """
    print('\nSecond Stage Starting...\n')
    train_probs, test_probs = CV(train_twoBBdf=train_ETs, test_twoBBdf=test_ETs, nfolds=5, random_seed=random_seed, array_index=True)
    train_probs = train_probs[train_probs>threshold] ; test_probs = test_probs[test_probs>threshold]
    print('\n\nSecond Stage Complete!!!\n\n')
    return train_probs, test_probs

def thirdStage(train_TAG_df, test_TAG_df,  train_TB_scores, test_TB_scores, train_path=train_path, test_path=test_path, random_seed=42):
    """This function does not take a twobbdf object but just a normal pandas dataframe which comes from combing the
       LOF_TBs and LOF_ETs dataframes. The preprocess4TAGGING function (found in utils) then prepares this dataframe for
       the tagging MVA which gives a tagging decision for each TB. Next we need to turn the TB tags into event tags, as
       some events have more than one TB
    """
    #prepare df for tagging
    print('\n\nStarting Third Stage...\n\n')
    ids = ['runNumber', 'eventNumber', 'nCandidate']
    train_TAG_df, train_event_ids, train_TAG_labels = preprocess4TAGGING(TAG_df=train_TAG_df, TB_scores=train_TB_scores, path=train_path)
    test_TAG_df, test_event_ids, test_TAG_labels = preprocess4TAGGING(TAG_df=test_TAG_df, TB_scores=test_TB_scores, path=test_path)

    #calculate tagging decisions for each TB
    train_TAG_probs, test_TAG_probs = CV(train_twoBBdf=train_TAG_df.drop(columns=ids + ['TB_id'], axis=0), test_twoBBdf=test_TAG_df.drop(columns=ids + ['TB_id'], axis=0),nfolds=5, random_seed=random_seed, justdf=True)

    #combine TB tag decisions into event tag decisions
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

