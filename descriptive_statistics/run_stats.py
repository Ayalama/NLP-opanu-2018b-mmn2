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
    total_unigrams = len(data_set.SEG)
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

#what is the differnace between this ans segment-tag instances?!
def count_segment_tag_types(data_set):
    return


# average number of tags appeared per segment
# referred keys in the data set are 'SEG' and 'TAG'
def index_of_ambiguity(data_set):
    df = data_set[['SEG', 'TAG']]
    df=df.drop_duplicates() # keep only uniq TAG+SEGMENT pairs
    df= df.groupby('SEG').agg('count').reset_index() # count number of uniq TAGS per segment
    return np.average(df.TAG) # return average of uniq tags per segment


if __name__ == '__main__':  # This is needed to allow multiprocessing in windows
    gold, train = read_train_gold()
    gold_unigrams_instances,gold_unigrams_rows = count_unigrams_instances(gold)
    gold_unigrams_types = count_unigrams_types(gold)
    gold_segment_tag_instances = count_segment_tag_instances(gold)
    gold_segment_tag_types = count_segment_tag_types(gold)
    gold_index_of_ambiguity = index_of_ambiguity(gold)
    print "Gold corpus statistics: \n" \
          "# number of unigram instances: {} \n" \
          "# number of unigram types: {} \n" \
          "# number of segment- tag instances: {} \n" \
          "# number of segment- tag types: {} \n" \
          "# index of ambiguity: {} \n".format(gold_unigrams_instances, gold_unigrams_types, gold_segment_tag_instances,
                                               gold_segment_tag_types, gold_index_of_ambiguity)

    train_unigrams_instances,train_unigrams_rows = count_unigrams_instances(train)
    train_unigrams_types = count_unigrams_types(train)
    train_segment_tag_instances = count_segment_tag_instances(train)
    train_segment_tag_types = count_segment_tag_types(train)
    train_index_of_ambiguity = index_of_ambiguity(train)
    print "Train corpus statistics: \n" \
          "# number of unigram instances: {} \n" \
          "# number of unigram types: {} \n" \
          "# number of segment- tag instances: {} \n" \
          "# number of segment- tag types: {} \n" \
          "# index of ambiguity: {} \n".format(train_unigrams_instances, train_unigrams_types,
                                               train_segment_tag_instances, train_segment_tag_types,
                                               train_index_of_ambiguity)

    all_unigrams_instances,train_unigrams_rows = count_unigrams_instances(pd.concat([train,gold]))
    all_unigrams_types = count_unigrams_types(pd.concat([train,gold]))
    all_segment_tag_instances = count_segment_tag_instances(pd.concat([train,gold]))
    all_segment_tag_types = count_segment_tag_types(pd.concat([train,gold]))
    all_index_of_ambiguity = index_of_ambiguity(pd.concat([train,gold]))
    print "Entire corpus statistics: \n" \
          "# number of unigram instances: {} \n" \
          "# number of unigram types: {} \n" \
          "# number of segment- tag instances: {} \n" \
          "# number of segment- tag types: {} \n" \
          "# index of ambiguity: {} \n".format(all_unigrams_instances, all_unigrams_types,
                                               all_segment_tag_instances, all_segment_tag_types,
                                               all_index_of_ambiguity)