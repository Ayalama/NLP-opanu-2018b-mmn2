import argparse
import taggers.basic_tagger as bstag
import taggers.first_ord_tagger as HMMtag
import taggers.first_ord_tagger_logprobs as HMMtag_logprob
import os.path

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='baseline',
                    help='name of model to be used for decoding. use one of "baseline" or "bi-gram"')

parser.add_argument('--test_file', type=str,
                    default='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\datasets\\heb-pos-small.test',
                    help='provide path for test file. format is SEG column, no header. sentences sepereted by an empty row.')

parser.add_argument('--param_file1', type=str, required=True,
                    help='specify path to param file for the model. for baseline model it''s common tags path and for bi-gram (or more), path to *.lex file')

parser.add_argument('--param_file2', type=str,
                    help='specify path to param file 2 for the model. relevant fo bi-gram or more model (path to *.gram file)')

args = parser.parse_args()

# args.test_file='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\datasets\\heb-pos.test'
print "executing decode for NLP model: model={model}, test file={test_file}".format(model=args.model,
                                                                                    test_file=args.test_file)
print "param files:"
print "param-file1={param_file1}".format(param_file1=args.param_file1)

if args.param_file2 is not None:
    print "param-file2={param_file2}".format(param_file2=args.param_file2)

if not os.path.isfile(args.test_file):
    raise Exception('No such file {}'.format(args.test_file))

if args.model == 'baseline':
    tagger = bstag.BasicTagger()
    tagger.decode(args.test_file, args.param_file1)  # input: test file and common tags file

if args.model == 'bi-gram':
    tagger = HMMtag.HMMTagger()
    tagger.decode(sen_file=args.test_file, lex_file=args.param_file1,
                  gram_file=args.param_file2, tagged_path='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\tests\\hmm_tagger.tagged')  # input: test file ,*.lex file, *.gram file

if args.model == 'bi-gram-logprob':
    tagger = HMMtag_logprob.HMMTagger_logprobs()
    tagger.decode(sen_file=args.test_file, lex_file=args.param_file1,
                  gram_file=args.param_file2,
                  tagged_path='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\tests\\hmm_tagger_run3\\hmm_tagger.tagged')
