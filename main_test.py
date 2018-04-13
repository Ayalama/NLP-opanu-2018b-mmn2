import taggers.basic_tagger as bstag
import taggers.first_ord_tagger as hmm
import evaluation.evaluation_measures as eval
import datasets.load_data_sets as ld

gold_file = r'C:\Users\aymann\PycharmProjects\maman_12_NLP\datasets\heb-pos.gold'
train_file = r'C:\Users\aymann\PycharmProjects\maman_12_NLP\datasets\heb-pos-small.train'
test_file = r'C:\Users\aymann\PycharmProjects\maman_12_NLP\datasets\heb-pos-small.test'


def test_basic_tagger(with_eval=False):
    tagbasic = bstag.BasicTagger()
    tagbasic.train(train_file)
    df=None
    if with_eval:
        tagbasic.evaluate(gold_file=gold_file,test_file=test_file,train_file=train_file)
    else:
        print "common value for tag: " + tagbasic.get_common_tag("CIBWR")
        df = tagbasic.decode(test_file)
    return df



def test_eval():
    gold_df = ld.load_gold_train(gold_file)
    gold_tagged_df = test_basic_tagger(gold_file)

    seg_acc_gld = eval.word_acc_for_sen(gold_df, gold_tagged_df)
    print "word_seg accuracy: " + str(float(sum(seg_acc_gld['SEN_WORD_ACC'])) / len(seg_acc_gld['SEN_WORD_ACC']))

    sen_acc_gld = eval.sentence_acc(gold_df, gold_tagged_df)
    print "word/seg accuracy: " + str(float(sum(sen_acc_gld['SEN_ACC'])) / len(sen_acc_gld['SEN_ACC']))

    corpus_wrd_acc = eval.word_acc_tst_corpuse(gold_df, gold_tagged_df)
    print "corpuse word accuracy: " + str(corpus_wrd_acc)

    corpus_sen_acc = eval.sentence_acc_tst_corpuse(gold_df, gold_tagged_df)
    print "corpuse sentence accuracy: " + str(corpus_sen_acc)


if __name__ == '__main__':  # This is needed to allow multiprocessing in windows
    # test_basic_tagger()
    hmmtag=hmm.HMMTagger()
    # hmmtag.train(train_file) # set train_data and lexical_data
    # df_to_decode=hmmtag.decode(test_file)
    gram_file='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\scripts\\hmm_train_test.gram'
    hmmtag.load_gram_from_file(gram_file)
    print ""