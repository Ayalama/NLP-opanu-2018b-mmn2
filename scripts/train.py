import argparse
import taggers.basic_tagger as bstag
import taggers.first_ord_tagger as HMMtag
import taggers.first_ord_tagger_logprobs as HMMTagger_logprobs
import os.path

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='baseline',
                    help='name of model to be used for training. use one of "baseline" or "bi-gram"')

parser.add_argument('--train_file', type=str, default='heb-pos.train',
                    help='provide path to training file. format is SEG,TAG columns, tab delimited, no header. sentences sepereted by an empty row.')

parser.add_argument('--smoothing', type=str, default='n',
                    help='specify if to use smoothing in the model. this param is insignificant for baseline model')

args = parser.parse_args()

if not os.path.isfile(args.train_file):
    raise Exception('No such file {}'.format(args.train_file))

# args.train_file='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\datasets\\heb-pos.train'
print "executing training phase for NLP model: model={model}, training file={training_file}, smoothing={smoothing}".format(
    model=args.model, training_file=args.train_file, smoothing=args.smoothing)

if args.model == 'baseline':
    tagger = bstag.BasicTagger()
    tagger.train(args.train_file, train_file_out=True)
if args.model == 'bi-gram':
    tagger = HMMtag.HMMTagger()
    lex_path_out = 'C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\tests\\hmm_tagger_run2\\hmm_tagger.lex'
    gram_path_out = 'C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\tests\\hmm_tagger_run2\\hmm_tagger.gram'
    tagger.train(args.train_file, train_file_out=True,lex_path_out=lex_path_out,gram_path_out=gram_path_out)


if args.model == 'bi-gram-logprob':
    tagger=HMMTagger_logprobs.HMMTagger_logprobs()
    lex_path_out = 'C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\tests\\hmm_tagger_run3\\hmm_tagger.lex'
    gram_path_out = 'C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\tests\\hmm_tagger_run3\\hmm_tagger.gram'
    tagger.train(args.train_file, train_file_out=True,lex_path_out=lex_path_out,gram_path_out=gram_path_out)
