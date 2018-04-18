import os.path
import sys

import src.taggers.basic_tagger as bstag
import src.taggers.first_ord_tagger_logprobs as HMMtag_logprob

# path to file tagged by model
model_tagged = sys.argv[1]
# provide path for gold file equivalent to tagged file. format is SEG column, TAG column, no header. sentences separated by an empty row.
gold_file = sys.argv[2]
# one of: baseline, bi-gram, bi-gram-logpro
model=  sys.argv[3]
# y/n
smoothing= sys.argv[4]




if not os.path.isfile(model_tagged):
    raise Exception('No such file {}'.format(model_tagged))
if not os.path.isfile(gold_file):
    raise Exception('No such file {}'.format(gold_file))


print "executing evaluation for NLP model: model={model}, model tagged file={model_tagged}, gold file={gold_file}, smoothing={smoothing}".format(
    model=model, model_tagged=model_tagged, gold_file=gold_file,smoothing=smoothing)

if model == '1':
    tagger = bstag.BasicTagger()
    tagger.evaluate(gold_file=gold_file, test_file=model_tagged,eval_out_file='basic_tagger.eval')  # input: test file and common tags file


if model == '2':
    tagger = HMMtag_logprob.HMMTagger_logprobs()
    tagger.evaluate(gold_file=gold_file, test_file=model_tagged, smoothing=smoothing,output_path='hmm_bigram_logprob_tagger.eval')  # input: test file ,*.lex file, *.gram file
