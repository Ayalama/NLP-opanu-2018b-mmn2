import pandas as pd
import datasets.load_data_sets as ld
import evaluation.evaluation_measures as eval
import scipy.stats


# train_file = "heb-pod.train"
class BasicTagger(object):
    def __init__(self):
        self.is_trained = False
        pass

    def train(self, train_file):
        self.train_data = ld.load_gold_train(train_file)
        self.train_seg_common = self.train_data.groupby('SEG')['TAG'].agg(
            lambda x: scipy.stats.mode(x)[0]).reset_index()
        self.is_trained = True
        return

    # def get_common_tag(self, segment):
    #     df = self.train_data.loc[self.train_data['SEG'] == segment]
    #     if len(df) == 0:
    #         return 'NNP'
    #     df = df[['SEG', 'TAG']]
    #     return df.mode().values[0][1]

    def get_common_tag(self, segment):
        df = self.train_seg_common[self.train_seg_common['SEG'] == segment]
        if len(df) == 0:
            return 'NNP'
        return df.values[0][1]

    def tag_sentances_file(self, sen_file_path):
        sen_df = ld.load_data(sen_file_path,
                              is_tagged=False)  # return data frame with columns ['SEG', 'SEN_NUM', 'WORD_NUM']
        sen_df['AUTO_TAG'] = 'NNP'
        # get common tag for each segment in the file
        for index, row in sen_df.iterrows():
            segment = row['SEG']
            sen_df.set_value(index, 'AUTO_TAG', self.get_common_tag(segment))
        return sen_df

    def evaluate(self, gold_file, test_file, train_file):
        if self.is_trained == False:
            self.train(train_file)

        gold_df = ld.load_gold_train(gold_file)
        test_tagged_df = self.tag_sentances_file(test_file)
        eval.output_eval('evaluation/basic_tagger_eval.txt', model_name="basic", test_file=test_file,
                         gold_file=gold_file,
                         gold_df=gold_df, test_tagged_df=test_tagged_df)
