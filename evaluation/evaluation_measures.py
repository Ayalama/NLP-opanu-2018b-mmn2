# TODO- implement methods below
import pandas as pd
import numpy as np


# proportion of correct tags in a sentence
# gold_df= file with answer manually decided. Tag column name should be 'TAG'
# gold_tagged_df= file with answer decided by our auto tagger. Tag column name should be 'AUTO_TAG'
# both data set should be annotated with sentence number and word number within the sentence
def word_acc_for_sen(gold_df, gold_tagged_df):
    df = get_correct_tags_cnt(gold_df, gold_tagged_df)
    df['SEN_WORD_ACC'] = df['CORR_YN'] / df['WORD_NUM']
    return df[['SEN_NUM','SEN_WORD_ACC']]


# whether or not all tags are correct in a sentence
def sentence_acc(gold_df, gold_tagged_df):
    df = get_correct_tags_cnt(gold_df, gold_tagged_df)
    df['SEN_ACC'] = np.where(df['CORR_YN'] == df['WORD_NUM'], 1, 0)
    return df[['SEN_NUM','SEN_ACC']]


# proportion of correct tags in the corpus
def word_acc_tst_corpuse(gold_df, gold_tagged_df):
    df=get_correct_tags_cnt(gold_df,gold_tagged_df)
    aj_df=word_acc_for_sen(gold_df,gold_tagged_df)
    df=pd.merge(df,aj_df,how='inner',on='SEN_NUM')
    corpuse_wrd_acc=sum(df['SEN_WORD_ACC']*df['WORD_NUM'])/sum(df['WORD_NUM'])
    return corpuse_wrd_acc


# proportion of correctly tagged sentences
def sentence_acc_tst_corpuse(gold_df, gold_tagged_df):
    df=sentence_acc(gold_df,gold_tagged_df)
    corpus_acc=sum(df['SEN_ACC'])/max(df['SEN_NUM'])
    return corpus_acc

def get_correct_tags_cnt(gold_df, gold_tagged_df):
    df = pd.merge(gold_df, gold_tagged_df, how='inner', on=['SEG', 'SEN_NUM', 'WORD_NUM'])
    df['CORR_YN'] = np.where(df['TAG'] == df['AUTO_TAG'], 1, 0)

    df_nj = df.groupby('SEN_NUM', as_index=False)['WORD_NUM'].max()  # get nj for each sentence
    df_eval_auto_tag = df.groupby('SEN_NUM', as_index=False)['CORR_YN'].sum()
    df_eval_auto_tag = pd.merge(df_eval_auto_tag, df_nj, how='inner', on='SEN_NUM')
    return df_eval_auto_tag
