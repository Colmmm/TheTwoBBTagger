from training import CV
import pandas as pd
from sklearn.metrics import accuracy_score
from utils import score_combiner, preprocess4TAGGING
train_path = '../TaggingJpsiK2012_fix_fix_1.root'
test_path = '../TaggingJpsiK2012_tiny_fix_fixSlice2.root'

def firstStage(train_TBs, test_TBs ,threshold, chunk_size, random_seed=42):
    """This function takes a TB's twobbdf object as its input, and then calculates the probabilities of how likely each
       TB comes from the decay of the Tagging B. These TBs are then filtered by getting rid of any TBs which have
       probabilities below the threshold (which is specified as an input parameter). This function then outputs a pandas
       series with a TBs index and the probabilities outputted by the TB MVA.
    """
    print('\nFirst Stage starting...\n\n')
    TB_train_df, TB_test_df, train_probs, test_probs,  = CV(train_twoBBdf=train_TBs, test_twoBBdf=test_TBs, nfolds=5, random_seed=random_seed, array_index=False, chunk_size=chunk_size )
    TB_train_df = TB_train_df.loc[train_probs>threshold] ; TB_test_df = TB_test_df.loc[test_probs>threshold]

    print('\n\nFirst Stage Complete!!!\n\n')
    return TB_train_df, TB_test_df, train_probs, test_probs

def secondStage(train_ETs, test_ETs, threshold, chunk_size, random_seed=42):
    """This function takes an ET twobbdf object as its input, and then calculates the probabilities of whether or not
       each ET comes from the decay of the Tagging B. Then similar to the TB case, the ETs are filtered by getting rid
       of any ET which has a probability below the threshold (which is specified as an input parameter). This function
       then outputs
     """
    print('\nSecond Stage Starting...\n')
    ET_train_df, ET_test_df, train_probs, test_probs = CV(train_twoBBdf=train_ETs, test_twoBBdf=test_ETs, nfolds=5, random_seed=random_seed, array_index=True, chunk_size=chunk_size)
    ET_train_df = ET_train_df[train_probs>threshold] ; ET_test_df = ET_test_df[test_probs>threshold]
    print('\n\nSecond Stage Complete!!!\n\n')
    return ET_train_df, ET_test_df

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
    train_TAG_probs, test_TAG_probs = CV(train_twoBBdf=train_TAG_df.drop(columns=ids + ['TB_id'], axis=0), test_twoBBdf=test_TAG_df.drop(columns=ids + ['TB_id'], axis=0),nfolds=5, random_seed=random_seed, justdf=True, chunk_size=None)

    #combine TB tag decisions into event tag decisions
    TAG_preds = pd.concat([test_TAG_probs, test_TB_scores, test_event_ids, test_TAG_labels.SignalB_ID], axis=1);
    TAG_preds.columns = ['TAG_scores', 'TB_scores', 'event_id', 'label']
    per_event_TAG = TAG_preds.groupby('event_id').apply(score_combiner)
    per_event_TAG = per_event_TAG.groupby('event_id').mean()

    per_event_TAG = per_event_TAG.drop(per_event_TAG.query('label!=0.0 and label!=1.0').index, axis=0)

    print('\n\nThird Stage Complete!!\n')
    print('\n\n')
    print(per_event_TAG.head())
    acc = accuracy_score(per_event_TAG.label, round(per_event_TAG.pred)) ; print('\nThe final Tagging Accuracy:\n{0}'.format(acc))
    tag_eff = 0.89*(per_event_TAG.shape[0]/1114) ; print('\nTagging efficiency:\n{0}'.format(tag_eff))
    print('\nOVERALL TAGGING POWER:\n{0}'.format(  tag_eff*(1-2*(1-acc))**2)  )

    return per_event_TAG

