import taggers.basic_tagger as bstag
import evaluation.evaluation_measures as eval
import datasets.load_data_sets as ld

gold_file = r'C:\Users\aymann\PycharmProjects\maman_12_NLP\datasets\heb-pos.gold'
train_file = r'C:\Users\aymann\PycharmProjects\maman_12_NLP\datasets\heb-pos.train'
test_file = r'C:\Users\aymann\PycharmProjects\maman_12_NLP\datasets\heb-pos.test'


def test_basic_tagger(test_set_file):
    tagbasic = bstag.BasicTagger()
    tagbasic.train(train_file)

    print "common value for tag: " + tagbasic.get_common_tag("CIBWR")
    df = tagbasic.tag_sentances_file(test_set_file)
    return df


if __name__ == '__main__':  # This is needed to allow multiprocessing in windows
    gold_df = ld.load_gold_train(gold_file)
    gold_tagged_df = test_basic_tagger(gold_file)
    # gold_tagged_df.rename(columns={1: 'TAG'}, inplace=True)
    print "corpuse word accuracy: "+ eval.word_acc_tst_corpuse(gold_df,gold_tagged_df)
    print "corpuse sentence accuracy: " + eval.sentence_acc_tst_corpuse(gold_df, gold_tagged_df)


