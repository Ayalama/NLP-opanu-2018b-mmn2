import pandas as pd
import numpy as np
import os
import os.path


# load heb-pos.gold file and return is in a vectorized format.
# structure of file should be seg<tab>tag<eol>.
# sentences are separated with an empty line
def load_gold_train(gold_or_train_file):
    return load_data(gold_or_train_file, is_tagged=True)


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
            seg_df.set_value(index, 'SEN_NUM', sen_num)
            seg_df.set_value(index, 'WORD_NUM', word_num)
            word_num = word_num + 1
        else:  # new sentence
            word_num = 1
            sen_num = sen_num + 1
    return seg_df[seg_df['SEG'] == seg_df['SEG']]


def split_to_fold_files(segments_file, num_folds):
    df = load_data(segments_file=segments_file, is_tagged=True)

    total_sentences = df.SEN_NUM.max()
    fold_size = int(total_sentences / num_folds)  # number of sentences in a fold

    folds_dir = os.path.dirname(segments_file)
    if not os.path.exists(folds_dir + "\\folds"):
        os.makedirs(folds_dir + "\\folds")

    folds_dir = folds_dir + "\\folds"
    folds_file_prefix = os.path.basename(segments_file).split('.')[0]

    sen_num = 1
    fold_itr = 1
    folds_paths_df = pd.DataFrame(columns=[['FOLD_NUM', 'FOLD_PATH']])
    while (sen_num < total_sentences + 1) and fold_itr < num_folds + 1:
        cur_fold_file_path = folds_dir + "\\" + folds_file_prefix + "_{}.fold".format(fold_itr)
        if total_sentences - (sen_num + fold_size) < fold_size:
            curr_fold_df = df[np.where((df['SEN_NUM'] >= sen_num), 1, 0) == 1]
        else:
            curr_fold_df = df[np.where((df['SEN_NUM'] >= sen_num) & (df['SEN_NUM'] < sen_num + fold_size), 1, 0) == 1]
        for i in xrange(sen_num, sen_num + fold_size):
            sen_segments = curr_fold_df[np.where(curr_fold_df['SEN_NUM'] == i, 1, 0) == 1]
            sen_segments = sen_segments[['SEG', 'TAG']]
            sen_segments = sen_segments.append({'SEG': '', 'TAG': ''}, ignore_index=True)
            sen_segments.to_csv(cur_fold_file_path, mode='a', sep='\t', header=False, index=False)

        folds_paths_df = folds_paths_df.append({'FOLD_NUM': fold_itr, 'FOLD_PATH': cur_fold_file_path},
                                               ignore_index=True)
        sen_num = sen_num + fold_size
        fold_itr += 1

    return folds_dir, folds_paths_df


def train_tagged_from_folds(folds_paths_df, min_foldid_train, max_foldid_train, out_dir=None):
    if out_dir is None:
        folds_dir = os.path.dirname(folds_paths_df.loc[[0]].FOLD_PATH[0])
        out_dir = folds_dir + "\\train_gold"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_out_path = out_dir + "\\heb-pos-folds_{min_foldid_train}_{max_foldid_train}.train".format(
        min_foldid_train=min_foldid_train, max_foldid_train=max_foldid_train)

    tagged_path = out_dir + "\\heb-pos-folds_{min_foldid_train}_{max_foldid_train}.tagged".format(
        min_foldid_train=min_foldid_train, max_foldid_train=max_foldid_train)

    train_f = open(train_out_path, "wb")

    for idx, fold_file in folds_paths_df.iterrows():
        if (max_foldid_train >= fold_file.FOLD_NUM >= min_foldid_train):
            with open(fold_file.FOLD_PATH, 'rb') as fold_train:
                train_f.write(fold_train.read())
            train_f.write("\n")

    train_f.close()
    return train_out_path, tagged_path


def prepare_all_folds_file(segments_file, num_folds):
    folds_dir, folds_paths_df = split_to_fold_files(segments_file=segments_file, num_folds=num_folds)

    folds_train_df = pd.DataFrame(columns=[['NUM_FOLD', 'TRAIN_PATH', 'TAGGED_PATH_TBD']])
    for i in xrange(1, num_folds + 1):
        train_out_path, tagged_path_tbd = train_tagged_from_folds(folds_paths_df=folds_paths_df,
                                                                  min_foldid_train=1,
                                                                  max_foldid_train=i)

        folds_train_df = folds_train_df.append(
            {'NUM_FOLD': i, 'TRAIN_PATH': train_out_path, 'TAGGED_PATH_TBD': tagged_path_tbd}, ignore_index=True)
    return folds_train_df


if __name__ == '__main__':  # This is needed to allow muliprocessing in windows
    print os.getcwd()
    segments_file = 'heb-pos.gold'
    golddf = load_gold_train(segments_file)
