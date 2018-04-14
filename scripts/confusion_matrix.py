import argparse
import os

import pandas as pd

import datasets.load_data_sets as ld
import evaluation.evaluation_measures as eval

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='baseline',
                    help='name of model to be used for decoding. use one of "baseline" or "bi-gram"')

parser.add_argument('--test_file', type=str,
                    default='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\datasets\\heb-pos-small.test',
                    help='path to training file"')
parser.add_argument('--gold_file', type=str,
                    default='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\datasets\\heb-pos-small.gold',
                    help='path to training file"')

parser.add_argument('--smoothing', type=str, default='n',
                    help='specify if to use smoothing in the model. this param is insignificant for baseline model')

args = parser.parse_args()

if not os.path.isfile(args.test_file):
    raise Exception('No such file {}'.format(args.test_file))
if not os.path.isfile(args.gold_file):
    raise Exception('No such file {}'.format(args.gold_file))

# args.test_file='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\datasets\\heb-pos.test'
print "building confusion matrix for NLP model: model={model}, model test_file={test_file},model gold_file={gold_file},smoothing={smoothing}".format(
    model=args.model, test_file=args.test_file, gold_file=args.gold_file,
    smoothing=args.smoothing)

confusion_out_path = "C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\tests\\confusion_matrix\\{}_model.confusion".format(args.model)
labals_out_path = "C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\tests\\confusion_matrix\\{}_model.labalidx".format(args.model)

test_df = ld.load_gold_train(args.test_file)
test_df.rename(columns={'TAG': 'AUTO_TAG'}, inplace=True)
gold_df = ld.load_gold_train(args.gold_file)

labels, confusion, confusion_df = eval.get_confusion_metrix(gold_df=gold_df, test_tagged_df=test_df)

pd.DataFrame(confusion).to_csv(confusion_out_path, sep='\t')
pd.DataFrame(labels).to_csv(labals_out_path, sep='\t', index=True,header=False)
print "full confusion metrix at {confusion_out_path}, with labels index at {labals_out_path}".format(
    confusion_out_path=confusion_out_path, labals_out_path=labals_out_path)

confusion_df=confusion_df.sort_values(by='CNT',ascending=False)
print "top 3 mistakes in confusion matrix:"
print confusion_df[0:3]

max_arg_gold = confusion_df[0:1].values[0][0]
max_arg_test = confusion_df[0:1].values[0][1]
max_value = int(confusion_df[0:1].values[0][2])

print "max confusion in model: model tagged {max_arg_test} where gold tag is {max_arg_gold} {max_value} times!".format(
    max_arg_test=max_arg_test, max_arg_gold=max_arg_gold, max_value=max_value)

