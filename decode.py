import os.path
import sys

import src.taggers.first_ord_tagger as HMMtag
import src.taggers.first_ord_tagger_logprobs as HMMtag_logprob
import src.taggers.basic_tagger as bstag

model = sys.argv[1]
test_file = sys.argv[2]
param_file1= sys.argv[3]

print "executing decode for NLP model: model={model}, test file={test_file}".format(model=model,test_file=test_file)
print "param files:"
print "param-file1={param_file1}".format(param_file1=param_file1)

if len(sys.argv>3):
    param_file2 = sys.argv[4]
    if param_file2 is not None:
        print "param-file2={param_file2}".format(param_file2=param_file2)

if not os.path.isfile(test_file):
    raise Exception('No such file {}'.format(test_file))

if model == 'baseline':
    tagger = bstag.BasicTagger()
    tagger.decode(test_file, param_file1,tagged_path='basic_tagger.tagged')  # input: test file and common tags file

if model == 'bi-gram' and param_file2 is not None:
    tagger = HMMtag.HMMTagger()
    tagger.decode(sen_file=test_file, lex_file=param_file1,
                  gram_file=param_file2, tagged_path='hmm_bigram_tagger.tagged')  # input: test file ,*.lex file, *.gram file

if model == 'bi-gram-logprob' and param_file2 is not None:
    tagger = HMMtag_logprob.HMMTagger_logprobs()
    tagger.decode(sen_file=test_file, lex_file=param_file1,
                  gram_file=param_file2,
                  tagged_path='hmm_bigram_logprob_tagger.tagged')
