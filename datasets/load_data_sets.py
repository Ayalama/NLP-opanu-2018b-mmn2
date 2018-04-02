import pandas as pd
import os


# load heb-pos.gold file and return is in a vectorized format.
# structure of file should be seg<tab>tag<eol>.
# sentences are separated with an empty line
def load_gold_train(gold_or_train_file):
    load_data(gold_or_train_file, is_tagged=True)


# load heb-pos.gold file and return is in a vectorized format.
# structure of file should be seg<eol>.
# sentences are separated with an empty line
# first column in ths file reserved for Segment
# if file has 'TAG' column than set is_tagged=True. In this case, the second column in the file will be considered as TAG
def load_data(segments_file, is_tagged=False):
    sen_num = 1
    word_num = 1

    seg_df = pd.read_csv(segments_file, sep='\t', header=None, skip_blank_lines=False)

    seg_df.rename(columns={0: 'SEG'}, inplace=True)
    if is_tagged:
        seg_df.rename(columns={1: 'TAG'}, inplace=True)

    seg_df['SEN_NUM'] = 0
    seg_df['WORD_NUM'] = 0

    for index, row in seg_df.iterrows():
        if row['SEG'] == row['SEG']:
            seg_df.set_value(index,'SEN_NUM',sen_num)
            seg_df.set_value(index,'WORD_NUM',word_num)
            word_num = word_num + 1
        else:  # new sentence
            word_num = 1
            sen_num = sen_num + 1
    return seg_df[seg_df['SEG'] == seg_df['SEG']]


if __name__ == '__main__':  # This is needed to allow muliprocessing in windows
    print os.getcwd()
    segments_file = 'heb-pos.gold'
    golddf = load_gold_train(segments_file)
