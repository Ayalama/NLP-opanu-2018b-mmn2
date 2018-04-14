import pandas as pd
import numpy as np
import datasets.load_data_sets as ld
import evaluation.evaluation_measures as eval
import scipy.stats
import os.path

# train_file = "heb-pod.train"
class BasicTagger(object):
    def __init__(self):
        self.is_trained = False
        pass

    def train(self, train_file, train_file_out=True, param_path=None):
        self.train_data = ld.load_gold_train(train_file)
        self.train_seg_common = self.train_data.groupby('SEG')['TAG'].agg(lambda x: scipy.stats.mode(x)[0]).reset_index()

        seg_tag_cnt = self.train_data.groupby(['SEG', 'TAG'], as_index=False)['WORD_NUM'].count()
        seg_tag_cnt.rename(columns={'WORD_NUM':'SEG_TAG_CNT'},inplace=True)
        seg_cnt = self.train_data.groupby('SEG', as_index=False)['WORD_NUM'].count()
        seg_cnt.rename(columns={'WORD_NUM': 'SEG_CNT'}, inplace=True)
        merged=pd.merge(self.train_seg_common,seg_tag_cnt, on=['SEG', 'TAG'])
        merged = pd.merge(merged, seg_cnt, on=['SEG'])
        merged['PROB']= merged['SEG_TAG_CNT']/merged['SEG_CNT']
        self.train_seg_common= merged[['SEG','TAG','PROB']]
        self.is_trained = True

        if train_file_out:
            if param_path is None:
                param_path=os.getcwd() + '\\basic_tagger_params.train'
            self.train_seg_common.to_csv(param_path, sep='\t', index=False)
            print "output basic_tagger_params.train file at: {}".format(param_path)
        return

    def get_common_tag(self, segment):
        df = self.train_seg_common[self.train_seg_common['SEG'] == segment]
        if len(df) == 0:
            return 'NNP'
        return df.values[0][1]

    def decode(self, sen_file_path, param_file=None,tagged_path=None):
        sen_df = ld.load_data(sen_file_path,
                              is_tagged=False)  # return data frame with columns ['SEG', 'SEN_NUM', 'WORD_NUM']
        sen_df['AUTO_TAG'] = 'NNP'

        if param_file is not None:
            self.load_train_seg_common(param_file)

        # get common tag for each segment in the file
        for index, row in sen_df.iterrows():
            segment = row['SEG']
            sen_df.set_value(index, 'AUTO_TAG', self.get_common_tag(segment))

        self.output_tagging(sen_df,tagged_path)
        return sen_df

    def evaluate(self, gold_file, test_file):
        gold_df = ld.load_gold_train(gold_file)
        test_tagged_df = ld.load_gold_train(test_file)
        output_path=os.getcwd() + '\\basic_tagger.eval'

        eval.output_eval(output_path, model_name="baseline", test_file=test_file,
                         gold_file=gold_file,
                         gold_df=gold_df, test_tagged_df=test_tagged_df)
        print "output basic_tagger evaluation file at: {}".format(output_path)

    def load_train_seg_common(self, param_file):
        self.train_seg_common= pd.read_csv(param_file, sep='\t')
        return

    def output_tagging(self, sen_df,tagged_path=None):
        if tagged_path is None:
            tagged_path = os.getcwd() + '\\basic_tagger.tagged'
        num_sentences=sen_df.SEN_NUM.max()

        for i in xrange(1,num_sentences+1):
            df=sen_df[np.where(sen_df['SEN_NUM']==i,1,0)==1][['SEG','AUTO_TAG']]
            df=df.append({'SEG':'','AUTO_TAG':''},ignore_index=True)
            df.to_csv(tagged_path, mode='a', sep='\t',header=False, index=False)

        print "output basic_tagger.tagged file at: {}".format(tagged_path)
        return
