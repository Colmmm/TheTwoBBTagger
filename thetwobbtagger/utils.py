def score_combiner(df):
    df['pred'] = sum(df.TAG_score * df.TB_score) / sum(df.TB_score)
    new_df = df[['pred', 'event_id', 'label']]
    return new_df