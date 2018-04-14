import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import datasets.load_data_sets as ld
import evaluation.evaluation_measures as eval
import taggers.basic_tagger as bstag
# import taggers.first_ord_tagger as HMMtag
import taggers.first_ord_tagger_logprobs as HMMtag

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='baseline',
                    help='name of model to be used for decoding. use one of "baseline" or "bi-gram"')

parser.add_argument('--train_file', type=str,
                    default='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\datasets\\heb-pos.train',
                    help='path to training file"')

parser.add_argument('--test_file', type=str,
                    default='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\datasets\\heb-pos.test',
                    help='path to training file"')
parser.add_argument('--gold_file', type=str,
                    default='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\datasets\\heb-pos.gold',
                    help='path to training file"')

parser.add_argument('--smoothing', type=str, default='n',
                    help='specify if to use smoothing in the model. this param is insignificant for baseline model')

args = parser.parse_args()

if not os.path.isfile(args.train_file):
    raise Exception('No such file {}'.format(args.train_file))
if not os.path.isfile(args.test_file):
    raise Exception('No such file {}'.format(args.train_file))
if not os.path.isfile(args.train_file):
    raise Exception('No such file {}'.format(args.train_file))

# args.test_file='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\datasets\\heb-pos.test'
print "executing cross validation for NLP model: model={model}, model train_file={train_file},model test_file={test_file},model gold_file={gold_file},smoothing={smoothing}".format(
    model=args.model, train_file=args.train_file,test_file=args.test_file,gold_file=args.gold_file, smoothing=args.smoothing)

num_folds=10
folds_train_gold_df=ld.prepare_all_folds_file(args.train_file, num_folds)
gold_df = ld.load_gold_train(args.gold_file)
x = np.zeros(10)
y = np.zeros(10)

if args.model == 'baseline':
    tagger = bstag.BasicTagger()
    for ix, test_i in folds_train_gold_df.iterrows():
        i=test_i.NUM_FOLD
        train_file=test_i.TRAIN_PATH
        out_tagged_file=test_i.TAGGED_PATH_TBD
        param_path=train_file.replace('.train','.trainout')

        tagger.train(train_file=train_file,train_file_out=False,param_path=param_path)
        tagger.decode(sen_file_path=args.test_file,tagged_path=out_tagged_file)

        test_tagged_df = ld.load_gold_train(out_tagged_file)
        test_tagged_df.rename(columns={'TAG':'AUTO_TAG'},inplace=True)

        x[i-1]=i
        y[i-1]=eval.word_acc_tst_corpuse(gold_df, test_tagged_df)

if args.model == 'bi-gram-logprob':
    tagger = HMMtag.HMMTagger_logprobs()
    print "starting iterations for model {}...".format(args.model)

    for ix, test_i in folds_train_gold_df.iterrows():
        print "starting itr {}...".format(test_i.NUM_FOLD)
        i=test_i.NUM_FOLD
        train_file=test_i.TRAIN_PATH
        out_tagged_file=test_i.TAGGED_PATH_TBD

        print "train model on train file {}...".format(train_file)
        lex_path=train_file.replace('.train','.lex')
        gram_path = train_file.replace('.train', '.gram')

        # tagger.train(train_file=train_file,train_file_out=False,lex_path_out=lex_path,gram_path_out=gram_path)
        tagger.train(train_file=train_file, train_file_out=False)

        print "ite {} decode...".format(test_i.NUM_FOLD)
        test_tagged_df = tagger.decode(sen_file=args.test_file,tagged_path=out_tagged_file)
        print "ite {} output tagged file at {}...".format(out_tagged_file)

        # test_tagged_df = ld.load_gold_train(out_tagged_file)
        # test_tagged_df.rename(columns={'TAG': 'AUTO_TAG'}, inplace=True)

        x[i-1]=i
        y[i-1]=eval.word_acc_tst_corpuse(gold_df, test_tagged_df)

plt.plot(x, y)
plt.show()
