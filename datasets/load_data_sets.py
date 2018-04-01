import pandas as pd
import os


# load heb-pos.gold file and return is in a vectorized format.
# structure of file should be seg<tab>tag<eol>.
# sentences are separated with an empty line
def load_gold_train(gold_or_train_file):
    # print os.getcwd()
    sen_num = 1
    word_num = 1

    gold = pd.read_csv(gold_or_train_file, sep='\t', header=None, skip_blank_lines=False,
                       names=['SEG', 'TAG', 'SEN_NUM', 'WORD_NUM'])
    gold['SEN_NUM'] = 0
    gold['WORD_NUM'] = 0

    for index, row in gold.iterrows():
        if row['SEG'] == row['SEG']:
            gold.set_value(index,'SEN_NUM',sen_num)
            gold.set_value(index,'WORD_NUM',word_num)
            word_num = word_num + 1
        else:  # new sentence
            word_num = 1
            sen_num = sen_num + 1
            # gold.drop(index,inplace=True)
    # gold= pd.read_csv(gold_file, sep='\t',header=None)
    return gold[gold['SEG'] == gold['SEG']]


# load heb-pos.gold file and return is in a vectorized format.
# structure of file should be seg<eol>.
# sentences are separated with an empty line
def load_test(test_file):
    return


if __name__ == '__main__':  # This is needed to allow muliprocessing in windows
    print os.getcwd()
    gold_or_train_file = 'heb-pos.gold'
    golddf = load_gold_train(gold_or_train_file)
    # print(golddf['SEG'][10:25])
