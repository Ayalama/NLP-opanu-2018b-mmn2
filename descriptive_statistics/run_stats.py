import datasets.load_data_sets as ld
import numpy as np
import pandas as pd


def read_train_gold():
    gold_set = ld.load_gold_train(r"C:\Users\aymann\PycharmProjects\maman_12_NLP\datasets\heb-pos.gold")
    train_set = ld.load_gold_train(r"C:\Users\aymann\PycharmProjects\maman_12_NLP\datasets\heb-pos.train")
    return gold_set, train_set


# count the number of unigram instances in the given data set
# referred key in the data set is 'SEG'
# returned number of unique unigrams and total unigrams
def count_unigrams_instances(data_set):
    uniq_unigrams = len(data_set.SEG.unique())
    total_unigrams = data_set.shape[0]
    return uniq_unigrams, total_unigrams


# count the number of unigram types in the given data set
# referred keys in the data set is 'SEG' and 'TAG' (as TYPE)
def count_unigrams_types(data_set):
    uniq_unigram_types = len(data_set.TAG.unique())
    return uniq_unigram_types


# count the number of segment-tag pairs in the given data set
# referred keys in the data set are 'SEG' and 'TAG'
def count_segment_tag_instances(data_set):
    instances = len(data_set.groupby(["SEG", "TAG"]))
    return instances


# count the number of segment-tag pairs in the given data set
# referred keys in the data set are 'SEG' and 'TAG'

# what is the differnace between this ans segment-tag instances?!
def count_segment_tag_types(data_set):
    return


# average number of tags appeared per segment
# referred keys in the data set are 'SEG' and 'TAG'
def index_of_ambiguity(data_set):
    df = data_set[['SEG', 'TAG']]
    df = df.drop_duplicates()  # keep only uniq TAG+SEGMENT pairs
    df = df.groupby('SEG').agg('count').reset_index()  # count number of uniq TAGS per segment
    return np.average(df.TAG)  # return average of uniq tags per segment


def print_stats(dataframe):
    unigrams_instances_uniq, unigrams_instances = count_unigrams_instances(dataframe)
    segment_tag_instances = count_segment_tag_instances(dataframe)
    unigrams_types = count_unigrams_types(dataframe)
    # segment_tag_types = count_segment_tag_types(dataframe)
    index_of_ambiguity_s = index_of_ambiguity(dataframe)
    print "# number of unigram instances: {unigram_sntances} \n" \
          "# number of unigram types: {unigram_types} \n" \
          "# number of segment- tag instances: {seg_tag_instance} \n" \
          "# number of segment- tag types: {seg_tag_typ} \n" \
          "# index of ambiguity: {amb_idx} \n".format(unigram_sntances=unigrams_instances,
                                                      unigram_types=unigrams_instances_uniq,
                                                      seg_tag_instance=segment_tag_instances,
                                                      seg_tag_typ=unigrams_types,
                                                      amb_idx=index_of_ambiguity_s)


if __name__ == '__main__':  # This is needed to allow multiprocessing in windows
    gold, train = read_train_gold()
    print "Gold corpus statistics: \n"
    print_stats(gold)

    print "Train corpus statistics: \n"
    print_stats(train)

    print "Entire corpus statistics: \n"
    print_stats(pd.concat([train, gold],ignore_index=True))
