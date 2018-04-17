import sys
import os.path

import src.taggers.first_ord_tagger as HMMtag
import src.taggers.first_ord_tagger_logprobs as HMMTagger_logprobs

import src.taggers.basic_tagger as bstag

# one of: baseline, bi-gram, bi-gram-logpro
model = sys.argv[1]
# provide path to training file. format is SEG,TAG columns, tab delimited, no header. sentences sepereted by an empty row
train_file = sys.argv[2]
# y/n
smoothing = sys.argv[3]

if not os.path.isfile(train_file):
    raise Exception('No such file {}'.format(train_file))

print "executing training phase for NLP model: model={model}, training file={training_file}, smoothing={smoothing}".format(
    model=model, training_file=train_file, smoothing=smoothing)

if model == 'baseline':
    tagger = bstag.BasicTagger()
    tagger.train(train_file, train_file_out=True, param_path='basic_tagger_params.train')
if model == 'bi-gram':
    tagger = HMMtag.HMMTagger()
    lex_path_out = 'hmm_bigram_tagger.lex'
    gram_path_out = 'hmm_bigram_tagger.gram'
    tagger.train(train_file, train_file_out=True, lex_path_out=lex_path_out, gram_path_out=gram_path_out)

if model == 'bi-gram-logprob':
    tagger = HMMTagger_logprobs.HMMTagger_logprobs()
    lex_path_out = 'hmm_bigram_logprob_tagger.lex'
    gram_path_out = 'hmm_bigram_logprob_tagger.gram'
    tagger.train(train_file, train_file_out=True, lex_path_out=lex_path_out, gram_path_out=gram_path_out)
