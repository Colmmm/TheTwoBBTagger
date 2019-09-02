def preprocess4TAGGING(TAG_df, TB_scores, path):
    """this function prepares the TAG_df for tagging, it does this by reading in the tagging labels (SignalB_ID) and also
       the ids and then joining them onto the TAG_df. This also replaces the labels from being +/-521 to 1/0. TB scores is
       also read in to filter out the bad TBs. Event ids and TAG_labels are also outputted as they will be used in combing
       of the TB tags into event tags"""
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

def score_combiner(df):
    """This function is used in the combing of the TB tags into event tags. I have to define a separate function for
       this, as this function is applied via a groupby apply. This function works on a groupby version of TBs and their
       tag estimates, the df rows are grouped by their events, and a new TAG prediction is calculated combing all the
        TB tags in a given event weighted by their TB scores from first stage"""
    df['pred'] = sum(df.TAG_scores * df.TB_scores) / sum(df.TB_scores)
    new_df = df[['pred', 'event_id', 'label']]
    return new_df


