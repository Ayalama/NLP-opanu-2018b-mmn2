import logging
import pandas as pd
import numpy as np


# proportion of correct tags in a sentence
# gold_df= file with answer manually decided. Tag column name should be 'TAG'
# gold_tagged_df= file with answer decided by our auto tagger. Tag column name should be 'AUTO_TAG'
# both data set should be annotated with sentence number and word number within the sentence
def word_acc_for_sen(gold_df, test_tagged_df):
    df = get_correct_tags_cnt(gold_df, test_tagged_df)
    df['SEN_WORD_ACC'] = df['CORR_YN'] / df['WORD_NUM']
    return df[['SEN_NUM', 'SEN_WORD_ACC']]


# whether or not all tags are correct in a sentence
def sentence_acc(gold_df, test_tagged_df):
    df = get_correct_tags_cnt(gold_df, test_tagged_df)
    df['SEN_ACC'] = np.where(df['CORR_YN'] == df['WORD_NUM'], 1, 0)
    return df[['SEN_NUM', 'SEN_ACC']]


# proportion of correct tags in the corpus
# test_tagged_df should have 'AUTO_TAG' column to compare with 'TAG' column of gold_df
def word_acc_tst_corpuse(gold_df, test_tagged_df):
    df = get_correct_tags_cnt(gold_df, test_tagged_df)
    aj_df = word_acc_for_sen(gold_df, test_tagged_df)
    df = pd.merge(df, aj_df, how='inner', on='SEN_NUM')
    corpuse_wrd_acc = float(sum(df['SEN_WORD_ACC'] * df['WORD_NUM'])) / sum(df['WORD_NUM'])
    return corpuse_wrd_acc


# proportion of correctly tagged sentences
def sentence_acc_tst_corpuse(gold_df, test_tagged_df):
    df = sentence_acc(gold_df, test_tagged_df)
    corpus_acc = float(sum(df['SEN_ACC'])) / max(df['SEN_NUM'])
    return corpus_acc


def get_correct_tags_cnt(gold_df, test_tagged_df):
    # assert len(gold_df) == len(test_tagged_df)
    df = pd.merge(gold_df, test_tagged_df, how='inner', on=['SEN_NUM', 'WORD_NUM'])
    df = df[['SEG_x','TAG','SEN_NUM','WORD_NUM','AUTO_TAG']]
    df = df.rename(columns={'SEG_x': 'SEG'})
    df['CORR_YN'] = np.where(df['TAG'] == df['AUTO_TAG'], 1, 0)

    df_nj = df.groupby('SEN_NUM', as_index=False)['WORD_NUM'].max()  # get nj for each sentence
    df_eval_auto_tag = df.groupby('SEN_NUM', as_index=False)['CORR_YN'].sum()
    df_eval_auto_tag = pd.merge(df_eval_auto_tag, df_nj, how='inner', on='SEN_NUM')
    return df_eval_auto_tag


def get_confusion_metrix(gold_df, test_tagged_df):
    assert len(gold_df) == len(test_tagged_df)
    df = pd.merge(gold_df, test_tagged_df, how='inner', on=['SEG', 'SEN_NUM', 'WORD_NUM'])
    labales = pd.unique(df['TAG'].append(df['AUTO_TAG'], ignore_index=True).values)
    df['CNT']=0
    confusion_df=df.groupby(['TAG','AUTO_TAG'], as_index=False)['CNT'].count()
    confusion_df=confusion_df[np.where(confusion_df['TAG']<>confusion_df['AUTO_TAG'], 1,0) == 1]

    number_cls = len(labales)
    logging.debug("Number of classes assumed to be {}".format(number_cls))

    confusion = np.zeros([number_cls, number_cls])
    # avoid the confusion with `0`
    # tran_pred = pred + 1
    for ix in xrange(number_cls):  # current class
        for iy in xrange(number_cls):
            cell_mtx = len(
                test_tagged_df[np.where((df['AUTO_TAG'] == labales[ix]) & (df['TAG'] == labales[iy]), 1, 0) == 1])
            confusion[ix, iy] = cell_mtx
    return labales,confusion,confusion_df


def output_eval(outputpath, model_name, test_file, gold_file, gold_df, test_tagged_df, smoothing='n'):
    eval_file = open(outputpath, "wb")

    eval_file.writelines("# Model: : %s \n" % model_name)
    eval_file.writelines("# Smoothing: : %s \n" % smoothing)
    eval_file.writelines("# Test File: : %s \n" % test_file)
    eval_file.writelines("# Gold File: : %s \n" % gold_file)
    eval_file.writelines('\n')

    eval_file.writelines('#######################################################\n')
    eval_file.writelines('# sent-num word-accuracy sent-accuracy\n')
    eval_file.writelines('#######################################################\n')

    test_tagged_df.rename(columns={'TAG':'AUTO_TAG'},inplace=True)

    accdf = pd.merge(word_acc_for_sen(gold_df, test_tagged_df), sentence_acc(gold_df, test_tagged_df), how='inner',
                     on='SEN_NUM')
    accdf.to_string(eval_file, index_names=False, header=False,index= False,col_space=0)
    eval_file.writelines('\n')
    eval_file.writelines('\n')

    eval_file.writelines('#######################################################\n')
    eval_file.writelines('# macro-avg < seg-accuracy-all > < sent-accuracy-all >\n')
    eval_file.writelines('#######################################################\n')

    seg_accuracy_all=word_acc_tst_corpuse(gold_df, test_tagged_df)
    sent_accuracy_all=sentence_acc_tst_corpuse(gold_df,test_tagged_df)
    line='macro-avg' + '\t' + str(seg_accuracy_all) + '\t' + str(sent_accuracy_all) + '\n' # TODO- what is macro-avg?!

    eval_file.writelines(line)

    eval_file.close()
