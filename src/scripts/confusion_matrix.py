import os.path
import sys

import pandas as pd

import src.datasets.load_data_sets as ld
import src.evaluation.evaluation_measures as eval

model = sys.argv[1]
test_file = sys.argv[2]
gold_file= sys.argv[3]

if not os.path.isfile(test_file):
    raise Exception('No such file {}'.format(test_file))
if not os.path.isfile(gold_file):
    raise Exception('No such file {}'.format(gold_file))

print "building confusion matrix for NLP model: model={model}, model test_file={test_file},model gold_file={gold_file}".format(
    model=model, test_file=test_file, gold_file=gold_file)

confusion_out_path = os.getcwd()+"\\{}_model.confusion".format(model)
labals_out_path = os.getcwd()+"\\{}_model.labalidx".format(model)

test_df = ld.load_gold_train(test_file)
test_df.rename(columns={'TAG': 'AUTO_TAG'}, inplace=True)
gold_df = ld.load_gold_train(gold_file)

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

