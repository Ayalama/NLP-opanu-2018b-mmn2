import argparse
import taggers.basic_tagger as bstag
import taggers.first_ord_tagger as HMMtag
import os.path

parser = argparse.ArgumentParser()

parser.add_argument('--model_tagged', type=str,
                    default='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\tests\\basic_tagger.tagged',
                    help='path to file tagged by model"')

parser.add_argument('--gold_file', type=str,
                    default='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\datasets\\heb-pos.gold',
                    help='provide path for gold file equivalent to tagged file. format is SEG column, TAG column, no header. sentences separated by an empty row.')

parser.add_argument('--model', type=str, default='baseline',
                    help='name of model to be used for decoding. use one of "baseline" or "bi-gram"')

parser.add_argument('--smoothing', type=str, default='n',
                    help='specify if to use smoothing in the model. this param is insignificant for baseline model')

args = parser.parse_args()

if not os.path.isfile(args.model_tagged):
    raise Exception('No such file {}'.format(args.test_file))
if not os.path.isfile(args.gold_file):
    raise Exception('No such file {}'.format(args.gold_file))

# args.test_file='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\datasets\\heb-pos.test'
print "executing evaluation for NLP model: model={model}, model tagged file={model_tagged}, gold file={gold_file}, smoothing={smoothing}".format(
    model=args.model, model_tagged=args.model_tagged, gold_file=args.gold_file,smoothing=args.smoothing)

if args.model == 'baseline':
    tagger = bstag.BasicTagger()
    tagger.evaluate(gold_file=args.gold_file, test_file=args.model_tagged)  # input: test file and common tags file

if args.model == 'bi-gram':
    tagger = HMMtag.HMMTagger()
    tagger.evaluate(gold_file=args.gold_file, test_file=args.model_tagged, smoothing=args.smoothing,output_path='C:\\Users\\aymann\\PycharmProjects\\maman_12_NLP\\tests\\hmm_tagger_run1\\hmm_tagger.eval')  # input: test file ,*.lex file, *.gram file
