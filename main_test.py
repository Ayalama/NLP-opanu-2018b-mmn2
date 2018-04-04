import taggers.basic_tagger as bstag
import evaluation.evaluation_measures as eval
import datasets.load_data_sets as ld

gold_file = r'C:\Users\aymann\PycharmProjects\maman_12_NLP\datasets\heb-pos.gold'
train_file = r'C:\Users\aymann\PycharmProjects\maman_12_NLP\datasets\heb-pos.train'
test_file = r'C:\Users\aymann\PycharmProjects\maman_12_NLP\datasets\heb-pos.test'


def test_basic_tagger(with_eval=False):
    tagbasic = bstag.BasicTagger()
    tagbasic.train(train_file)
    if with_eval:
        tagbasic.evaluate(gold_file=gold_file,test_file=test_file,train_file=train_file)
    else:
        print "common value for tag: " + tagbasic.get_common_tag("CIBWR")
        df = tagbasic.tag_sentances_file(test_file)
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
    #TODO- check evaluation file output!!
    test_basic_tagger(with_eval=True)

    print ""