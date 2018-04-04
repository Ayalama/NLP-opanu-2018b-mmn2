import datasets.load_data_sets as ld
import numpy as np
import pandas as pd


# train_file = "heb-pod.train"


class MarkovTagger(object):
    def __init__(self):
        pass

    def train(self, train_file):
        self.train_data = ld.load_gold_train(train_file)

        return

    # create lexical *.lex file containing row for each SEG
    # for each SEG-TAG, contain transitions probabilities between states
    def set_lexical(self):
        # columns ['SEG','TAG', 'SEN_NUM', 'WORD_NUM']

        df_seg_tag_cnt = self.train_data.groupby(['SEG', 'TAG'], as_index=False)['WORD_NUM'].count().to_frame(
            'SEG_TAG_CNT')  # WORD_NUM will be the number of instances for TAG, for this Segment
        df_seg_cnt = self.train_data.groupby('SEG', as_index=False).count().to_frame('SEG_CNT')  #
        df_seg = pd.merge(df_seg_tag_cnt, df_seg_cnt, how='inner', on='SEG')
        df_seg['TAG_PROB'] = df_seg['SEG_TAG_CNT'] / df_seg['SEG_CNT'] #TODO- change to log prob!!!

        # lexical=df_seg.groupby('SEG')[['TAG','TAG_PROB']].apply(lambda x: '\t'.join(x) )
        # self.lexical = df_seg.groupby('SEG')[['TAG', 'TAG_PROB']].apply(list)
        self.lexical = df_seg[['SEG', 'TAG', 'TAG_PROB']]

        return

    # output to file (TAGS and probs seperated by tabs
    def create_lexical(self):
        self.set_lexical()
        rows = pd.unique(self.lexical['SEG'].values)
        rows.reset_index()
        columns = pd.unique(self.lexical['TAG'].values)
        columns.reset_index()
        lex = np.zeros([len(rows), len(columns)])

        for ix, seg in pd.itterrows(rows):  # current class
            for iy, tag in pd.itterrows(columns):
                prob = self.lexical[self.lexical['SEG'] == seg.values[0] & self.lexical['TAG'] == tag.values[0]].values[0]
                tag_str=tag.values[0]
                lex[ix, iy] = tag_str+'\t'+prob
        return rows,columns,lex

    # create *.gram file
    # probability to move from one segment to another
    # TODO
    def create_gram(self):
        return

    # input- untagged sentence file (formatted as test
    # output- tagged sentence file/DF according to training. formatted as gold file
    # TODO
    def decode(self, sen_file):
        return
