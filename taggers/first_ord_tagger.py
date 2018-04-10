import datasets.load_data_sets as ld
import numpy as np
import pandas as pd


# train_file = "heb-pod.train"


class HMMTagger(object):
    def __init__(self):
        self.is_lex_set = False
        self.is_trained = False
        pass

    def train(self, train_file):
        self.train_data = ld.load_gold_train(train_file)
        self.set_lexical()
        return

    # create lexical *.lex file containing row for each SEG
    # for each SEG-TAG, contain transitions probabilities between states
    def set_lexical(self):
        # columns ['SEG','TAG', 'SEN_NUM', 'WORD_NUM']

        df_seg_tag_cnt = self.train_data.groupby(['SEG', 'TAG'], as_index=False)[
            'WORD_NUM'].count().reset_index()  # WORD_NUM will be the number of instances for TAG, for this Segment
        df_seg_tag_cnt.rename(index=str, columns={'WORD_NUM': 'SEG_TAG_CNT'}, inplace=True)
        df_seg_cnt = self.train_data.groupby('SEG', as_index=False)['WORD_NUM'].count().reset_index()
        df_seg_cnt.rename(index=str, columns={'WORD_NUM': 'SEG_CNT'}, inplace=True)
        df_seg = pd.merge(df_seg_tag_cnt, df_seg_cnt, how='inner', on='SEG')
        df_seg['TAG_PROB'] = df_seg['SEG_TAG_CNT'] / df_seg['SEG_CNT']  # TODO- change to log prob!!!

        # lexical=df_seg.groupby('SEG')[['TAG','TAG_PROB']].apply(lambda x: '\t'.join(x) )
        # self.lexical = df_seg.groupby('SEG')[['TAG', 'TAG_PROB']].apply(list)
        self.lexical = df_seg[['SEG', 'TAG', 'TAG_PROB']]
        self.is_lex_set = True

        return

    # output to file (TAGS and probs seperated by tabs
    def create_lexical(self):
        if not self.is_lex_set:
            self.set_lexical()

        rows = self.lexical['SEG'].drop_duplicates().reset_index()
        columns = self.lexical['TAG'].drop_duplicates().reset_index()
        lex = np.empty([len(rows), len(columns)], dtype=object)

        for ix, seg in rows.iterrows():  # current class
            for iy, tag in columns.iterrows():
                seg_tag_df = self.lexical[
                    np.where((self.lexical['SEG'] == seg.values[1]) & (self.lexical['TAG'] == tag.values[1]), 1,
                             0) == 1]
                if len(seg_tag_df) > 0:
                    prob = self.lexical[
                        np.where((self.lexical['SEG'] == seg.values[1]) & (self.lexical['TAG'] == tag.values[1]), 1,
                                 0) == 1].values[0][2]
                else:
                    prob = 0
                prob_str = str(prob)
                tag_str = tag.values[1]
                lex[ix, iy] = tag_str + '\t' + prob_str
        return rows, columns, lex

    # create *.gram file
    # probability to move from one tag to another
    # pairs of ti-t(i-1) and their probabiities (transition_prob)
    # TODO- test
    def create_gram(self):
        tags = self.train_data[['TAG', 'SEN_NUM', 'WORD_NUM']].drop_duplicates(
            subset=['TAG', 'SEN_NUM', 'WORD_NUM']).reset_index()
        tags_pairs = pd.DataFrame(columns=['TAG_i', 'TAG_i-1'])
        for ix, tagi_1 in tags.iterrows():  # current class
            next_tags = tags[
                np.where((tags['SEN_NUM'] == tagi_1.values[1]) & (self.lexical['WORD_NUM'] == tagi_1.values[2] + 1), 1,
                         0) == 1]  # all tags that apeared after tagi_1 (same sentence, one word after it)
            next_tags = next_tags[['TAG']].rename(columns={"TAG": "TAG_i"})
            next_tags['TAG_i-1'] = tagi_1.values[0]
            if next_tags is not None:
                tags_pairs.append(next_tags)

        df_tag_cnt = tags.groupby(['TAG'], as_index=False)['WORD_NUM'].count().reset_index()
        df_tag_cnt.rename(index=str, columns={'WORD_NUM': 'TAG_FRQ_CNT', 'TAG': 'TAG_i'}, inplace=True)
        df_pairs_cnt = tags_pairs.groupby(['TAG_i', 'TAG_i-1'], as_index=False)['WORD_NUM'].count().reset_index()
        df_pairs_cnt.rename(index=str, columns={'WORD_NUM': 'PAIR_FREQ_CNT'}, inplace=True)
        df_merge = pd.merge(df_tag_cnt, df_pairs_cnt, how='inner', on='TAG_i')
        df_merge['PROB'] = df_merge['PAIR_FREQ_CNT']/df_merge['TAG_FREQ_CNT']
         # df_merge.apply(lambda row: (row.PAIR_FREQ_CNT/row.TAG_FREQ_CNT), axis=1)
        self.transition_probs = df_merge[['TAG_i', 'TAG_i-1','PROB']]


    # input- untagged sentence file (formatted as test
    # output- tagged sentence file/DF according to training. formatted as gold file
    # TODO
    def decode(self, sen_file):
        return

    def evaluate(self, gold_file, test_file, train_file):
        if self.is_trained == False:
            self.train(train_file)

        gold_df = ld.load_gold_train(gold_file)
        test_tagged_df = self.decode(test_file)
        eval.output_eval('evaluation/hmm_tagger_eval.txt', model_name="HMM", test_file=test_file, gold_file=gold_file,
                         gold_df=gold_df, test_tagged_df=test_tagged_df)
