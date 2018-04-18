import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import src.datasets.load_data_sets as ld
import src.evaluation.evaluation_measures as eval
import src.taggers.basic_tagger as bstag
import src.taggers.first_ord_tagger_logprobs as HMMtag

model = sys.argv[1]
train_file = sys.argv[2]
test_file = sys.argv[3]
gold_file = sys.argv[4]

if not os.path.isfile(train_file):
    raise Exception('No such file {}'.format(train_file))
if not os.path.isfile(test_file):
    raise Exception('No such file {}'.format(test_file))
if not os.path.isfile(gold_file):
    raise Exception('No such file {}'.format(gold_file))

print "executing cross validation for NLP model: model={model}, model train_file={train_file},model test_file={test_file},model gold_file={gold_file}}".format(
    model=model, train_file=train_file, test_file=test_file, gold_file=gold_file)

num_folds = 10
folds_train_gold_df = ld.prepare_all_folds_file(train_file, num_folds)
gold_df = ld.load_gold_train(gold_file)
x = np.zeros(10)
y = np.zeros(10)

# 'baseline'
if model == '1':
    tagger = bstag.BasicTagger()
    for ix, test_i in folds_train_gold_df.iterrows():
        i = test_i.NUM_FOLD
        train_file = test_i.TRAIN_PATH
        out_tagged_file = test_i.TAGGED_PATH_TBD
        param_path = train_file.replace('.train', '.trainout')

        tagger.train(train_file=train_file, train_file_out=False, param_path=param_path)
        tagger.decode(sen_file_path=test_file, tagged_path=out_tagged_file)

        test_tagged_df = ld.load_gold_train(out_tagged_file)
        test_tagged_df.rename(columns={'TAG': 'AUTO_TAG'}, inplace=True)

        x[i - 1] = i
        y[i - 1] = eval.word_acc_tst_corpuse(gold_df, test_tagged_df)

# 'bi-gram-logprob'
if model == '2':
    tagger = HMMtag.HMMTagger_logprobs()
    print "starting iterations for model {}...".format(model)

    for ix, test_i in folds_train_gold_df.iterrows():
        print "starting itr {}...".format(test_i.NUM_FOLD)
        i = test_i.NUM_FOLD
        train_file = test_i.TRAIN_PATH
        out_tagged_file = test_i.TAGGED_PATH_TBD

        print "train model on train file {}...".format(train_file)
        lex_path = train_file.replace('.train', '.lex')
        gram_path = train_file.replace('.train', '.gram')

        tagger.train(train_file=train_file, train_file_out=False)

        print "ite {} decode...".format(test_i.NUM_FOLD)
        test_tagged_df = tagger.decode(sen_file=test_file, tagged_path=out_tagged_file)
        print "ite {} output tagged file at {}...".format(test_i.TRAIN_PATH, out_tagged_file)

        x[i - 1] = i
        y[i - 1] = eval.word_acc_tst_corpuse(gold_df, test_tagged_df)

plt.plot(x, y)
plt.title("Model learning curve")
plt.ylabel("accuracy")
plt.ylabel("iteration number")
plt.show()
