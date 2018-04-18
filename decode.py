import os.path
import sys

import src.taggers.basic_tagger as bstag
import src.taggers.first_ord_tagger_logprobs as HMMtag_logprob

model = sys.argv[1]
test_file = sys.argv[2]
param_file1= sys.argv[3]
param_file2=None

print "executing decode for NLP model: model={model}, test file={test_file}".format(model=model,test_file=test_file)
print "param files:"
print "param-file1={param_file1}".format(param_file1=param_file1)

if len(sys.argv)>4:
    param_file2 = sys.argv[4]
    if param_file2 is not None:
        print "param-file2={param_file2}".format(param_file2=param_file2)

if not os.path.isfile(test_file):
    raise Exception('No such file {}'.format(test_file))

# 'baseline'
if model == '1':
    print "start decode using basic tagger..."
    tagger = bstag.BasicTagger()
    tagger.decode(test_file, param_file1,tagged_path=os.getcwd()+'\\basic_tagger.tagged')  # input: test file and common tags file

# bi-gram-logprob
if model == '2' and param_file2 is not None:
    print "start decode using first order tagger..."
    tagger = HMMtag_logprob.HMMTagger_logprobs()
    tagger.decode(sen_file=test_file, lex_file=param_file1,
                  gram_file=param_file2,
                  tagged_path=os.getcwd()+'\\hmm_bigram_logprob_tagger.tagged')
