"""Python file containing utility functions for all sorts of different things"""


def score_combiner(df):

    df['pred'] = sum(df.TAG_scores * df.TB_scores) / sum(df.TB_scores)
    new_df = df[['pred', 'event_id', 'label']]
    return new_df