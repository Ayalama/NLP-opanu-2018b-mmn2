import datasets.load_data_sets as ld
import numpy as np
import pandas as pd
import os
import csv


# train_file = "heb-pod.train"


class HMMTagger(object):
    def __init__(self):
        self.train_data = None
        self.states = None
        self.lexical = None
        self.transition_probs = None
        pass

    def train(self, train_file, train_file_out=True):
        self.set_train_data(train_file)
        self.set_lexical()
        self.set_transition_probs()
        self.set_states()

        if train_file_out:
            self.output_lex()
            self.output_gram()
            # (os.getcwd() + '\\tests\\basic_tagger_params.train', sep='\t', index=False)
        return

    def set_states(self):
        self.states = self.lexical[['TAG']].drop_duplicates()
        self.states = self.states.append({'TAG': 'START'}, ignore_index=True)
        self.states = self.states.append({'TAG': 'END'}, ignore_index=True)

    def set_train_data(self, train_file):
        self.train_data = ld.load_gold_train(train_file)
        self.train_data = self.set_EOS(self.train_data)
        return

    def set_EOS(self, tags_df):
        max_wordnum = tags_df.groupby(['SEN_NUM'], as_index=False)['WORD_NUM'].max()
        max_wordnum.rename(index=str, columns={'WORD_NUM': 'MAX_WORD_NUM'}, inplace=True)
        new_tags_df = pd.merge(tags_df, max_wordnum, how='inner', on='SEN_NUM')
        new_tags_df['EOS'] = np.where((new_tags_df['MAX_WORD_NUM'] == new_tags_df['WORD_NUM']), 1, 0)
        return new_tags_df

    # create lexical *.lex file containing row for each SEG
    # for each SEG-TAG, contain transitions probabilities between states
    def set_lexical(self):
        # columns ['SEG','TAG', 'SEN_NUM', 'WORD_NUM']
        df_seg_tag_cnt = self.train_data.groupby(['SEG', 'TAG'], as_index=False)[
            'WORD_NUM'].count().reset_index()  # WORD_NUM will be the number of instances for TAG, for this Segment
        df_seg_tag_cnt.rename(index=str, columns={'WORD_NUM': 'SEG_TAG_CNT'}, inplace=True)
        df_tag_cnt = self.train_data.groupby('TAG', as_index=False)['WORD_NUM'].count().reset_index()
        df_tag_cnt.rename(index=str, columns={'WORD_NUM': 'TAG_CNT'}, inplace=True)
        df_seg = pd.merge(df_seg_tag_cnt, df_tag_cnt, how='inner', on='TAG')
        df_seg['SEG_PROB'] = df_seg['SEG_TAG_CNT'] / df_seg['TAG_CNT']  # TODO- change to log prob!!!
        self.lexical = df_seg[['SEG', 'TAG', 'SEG_PROB']]

        return

    # output to file (TAGS and probs separated by tabs)
    def create_lexical_format(self):
        if self.lexical is None:
            self.set_lexical()

        rows_segments = self.lexical['SEG'].drop_duplicates().reset_index()
        columns_tags = self.lexical['TAG'].drop_duplicates().reset_index()
        columns_tags.index += 1
        lex = np.empty([len(rows_segments), len(columns_tags) + 1], dtype=object)

        for ix, seg in rows_segments.iterrows():  # current class
            lex[ix, 0] = seg.values[1]
            for iy, tag in columns_tags.iterrows():
                seg_tag_df = self.lexical[
                    np.where((self.lexical['SEG'] == seg.values[1]) & (self.lexical['TAG'] == tag.values[1]), 1,
                             0) == 1]
                if len(seg_tag_df) > 0:
                    prob = self.lexical[
                        np.where((self.lexical['SEG'] == seg.values[1]) & (self.lexical['TAG'] == tag.values[1]), 1,
                                 0) == 1].values[0][2]
                else:
                    prob = 0
                prob_str = str(prob)
                tag_str = tag.values[1]
                lex[ix, iy] = tag_str + '\t' + prob_str
        return rows_segments, columns_tags, lex

    # create *.gram file
    # probability to move from one tag to another
    # pairs of ti-t(i-1) and their probabilities (transition_prob)
    def set_transition_probs(self):
        if self.train_data is None or self.lexical is None:
            print "training data is None"
            return

        tags = self.train_data[['TAG', 'SEN_NUM', 'WORD_NUM', 'EOS']].drop_duplicates(
            subset=['TAG', 'SEN_NUM', 'WORD_NUM', 'EOS'])
        tags_pairs = pd.DataFrame(columns=['TAG_i', 'TAG_i-1'])
        for ix, tagi_1 in tags.iterrows():  # current class

            next_tags = tags[
                np.where((tags['SEN_NUM'] == tagi_1.SEN_NUM) & (tags['WORD_NUM'] == tagi_1.WORD_NUM + 1), 1,
                         0) == 1]  # all tags that apeared after tagi_1 (same sentence, one word after it)
            next_tags = next_tags[['TAG']].rename(columns={"TAG": "TAG_i"})
            next_tags['TAG_i-1'] = tagi_1.TAG
            if next_tags is not None:
                tags_pairs = tags_pairs.append(next_tags, ignore_index=True)
            if tagi_1.WORD_NUM == 1:  # add pair instance of start-tagi
                tags_pairs = tags_pairs.append({'TAG_i-1': 'START', 'TAG_i': tagi_1.TAG}, ignore_index=True)
            if tagi_1.EOS == 1:  # add pair instance of tagi-END
                tags_pairs = tags_pairs.append({'TAG_i-1': tagi_1.TAG, 'TAG_i': 'END'}, ignore_index=True)

        df_tag_cnt = tags_pairs.groupby(['TAG_i-1'], as_index=False)['TAG_i'].count()
        df_tag_cnt.rename(index=str, columns={'TAG_i': 'TAG_FREQ_CNT'}, inplace=True)

        tags_pairs['PAIR_FREQ_CNT'] = 0
        df_pairs_cnt = tags_pairs.groupby(['TAG_i', 'TAG_i-1'], as_index=False)['PAIR_FREQ_CNT'].count()
        df_merge = pd.merge(df_tag_cnt, df_pairs_cnt, how='inner', on='TAG_i-1')
        df_merge['PROB'] = df_merge['PAIR_FREQ_CNT'] / df_merge['TAG_FREQ_CNT']
        self.transition_probs = df_merge[['TAG_i', 'TAG_i-1', 'PROB']]

    # input- untagged sentence file (formatted as test
    # output- tagged sentence file/DF according to training. formatted as gold file
    # TODO- check correctness
    def decode(self, sen_file, lex_file=None, gram_file=None):
        df_to_decode = ld.load_data(sen_file, is_tagged=False)
        df_to_decode = self.set_EOS(df_to_decode)
        decoded_tags = pd.DataFrame(columns=['SEN_NUM', 'WORD_NUM', 'TAG'])

        if lex_file is not None:
            self.load_lex_from_file(lex_file)
        if gram_file is not None:
            self.load_gram_from_file(gram_file)
        self.set_states()

        df_v = pd.DataFrame(columns=['step_idx', 'state', 'val'])
        df_b = pd.DataFrame(columns=['step_idx', 'state', 'prev_state'])

        # loop on all sentences
        for i in xrange(1, df_to_decode.SEN_NUM.max() + 1):
            sentence_words_df = df_to_decode[np.where((df_to_decode['SEN_NUM'] == i), 1, 0) == 1]
            words_array = sentence_words_df.as_matrix(columns=['SEG']).flat
            # initialize
            df_v_tmp, df_b_tmp = self.viterbi_init(words_array)
            df_v = df_v.append(df_v_tmp)
            df_b = df_b.append(df_b_tmp)
            # recursion
            decoded_sen_tags = self.viterbi_recursion(words_array, df_v, df_b)
            decoded_sen_tags['SEN_NUM'] = i

            decoded_tags = decoded_tags.append(decoded_sen_tags, ignore_index=True)
            df_v = pd.DataFrame(columns=['step_idx', 'state', 'val'])
            df_b = pd.DataFrame(columns=['step_idx', 'state', 'prev_state'])

        df_to_decode = np.merge(df_to_decode, decoded_tags, on=['SEN_NUM', 'WORD_NUM'])
        self.output_tagging(df_to_decode)
        return df_to_decode

    # sentence= array of words w1,...wn
    def viterbi_init(self, sentence):
        w1 = sentence[0]
        df_v = pd.DataFrame(columns=['step_idx', 'state', 'val'])
        df_b = pd.DataFrame(columns=['step_idx', 'state', 'prev_state'])

        for ix, s in self.states.iterrows():
            step_idx = 1
            state_name = s.TAG
            if not (state_name == 'START' or state_name == 'END'):
                p_s_start = self.transition_probs[np.where(
                    (self.transition_probs['TAG_i-1'] == 'START') & (self.transition_probs['TAG_i'] == state_name), 1,
                    0) == 1].PROB
                if len(p_s_start) == 0:
                    p_s_start = 0
                else:
                    p_s_start = p_s_start.values[0]

                p_w1_s = self.lexical[
                    np.where((self.lexical['SEG'] == w1) & (self.lexical['TAG'] == state_name), 1, 0) == 1].SEG_PROB
                if len(p_w1_s) == 0:
                    p_w1_s = 0
                else:
                    p_w1_s = p_w1_s.values[0]

                val = p_s_start * p_w1_s
                df_v = df_v.append({'step_idx': step_idx, 'state': state_name, 'val': val}, ignore_index=True)
                df_b = df_b.append({'step_idx': step_idx, 'state': state_name, 'prev_state': 'START'},
                                   ignore_index=True)
        return df_v, df_b

    def viterbi_recursion(self, sentence, df_v, df_b):
        num_words = len(sentence) + 1
        for i in xrange(2, num_words):
            wi = sentence[i - 1]
            step_idx = i
            prev_df_v = df_v[np.where(df_v['step_idx'] == i - 1, 1,
                                      0) == 1]  # all tags that are prior to current tag and their probs in input sentence
            if self.is_word_known(wi):
                for ix, s in self.states[np.where((self.states['TAG'] <> 'START') & (self.states['TAG'] <> 'END'), 1,
                                                  0) == 1].iterrows():
                    cur_state_name = s.TAG
                    p_wi_s = self.lexical[
                        np.where((self.lexical['SEG'] == wi) & (self.lexical['TAG'] == cur_state_name), 1,
                                 0) == 1].SEG_PROB
                    if len(p_wi_s) == 0:
                        p_wi_s = 0
                    else:
                        p_wi_s = p_wi_s.values[0]

                    if len(prev_df_v) == 1 and prev_df_v.state.values[0] == 'NNP':
                        v_i_s = 1 * p_wi_s
                        b_i_s = 'NNP'
                    else:
                        df_p_s_prvs = self.transition_probs[
                            np.where(self.transition_probs['TAG_i'] == cur_state_name, 1,
                                     0) == 1]  # all tags that are prior to current tag and their probs in training set
                        df_p_s_prvs_v = pd.merge(prev_df_v, df_p_s_prvs, left_on='state', right_on='TAG_i-1')
                        df_p_s_prvs_v['val_new'] = df_p_s_prvs_v['val'] * df_p_s_prvs_v['PROB'] * p_wi_s
                        v_i_s = df_p_s_prvs_v.val_new.max()
                        b_i_s = df_p_s_prvs_v.loc[[df_p_s_prvs_v.val_new.idxmax()]]['TAG_i-1'].values[0]

                    df_v = df_v.append({'step_idx': step_idx, 'state': cur_state_name, 'val': v_i_s}, ignore_index=True)
                    df_b = df_b.append({'step_idx': step_idx, 'state': cur_state_name, 'prev_state': b_i_s},
                                       ignore_index=True)
            else:
                cur_state_name = 'NNP'
                prev_df_v = df_v[np.where(df_v['step_idx'] == i - 1, 1, 0) == 1]
                b_i_s = prev_df_v.loc[[prev_df_v.val.idxmax()]].state.values[0]
                df_v = df_v.append({'step_idx': step_idx, 'state': cur_state_name, 'val': 1}, ignore_index=True)
                df_b = df_b.append({'step_idx': step_idx, 'state': cur_state_name, 'prev_state': b_i_s},
                                   ignore_index=True)

        # add 'END' state lines
        prev_df_v = df_v[np.where(df_v['step_idx'] == num_words - 1, 1, 0) == 1]
        df_p_s_prvs = self.transition_probs[
            np.where(self.transition_probs['TAG_i'] == 'END', 1, 0) == 1]  # all tags arrived before END
        df_p_s_prvs_v = pd.merge(prev_df_v, df_p_s_prvs, left_on='state', right_on='TAG_i-1')
        df_p_s_prvs_v['val_new'] = df_p_s_prvs_v['val'] * df_p_s_prvs_v['PROB']
        v_end = df_p_s_prvs_v.val_new.max()
        b_end = df_p_s_prvs_v.loc[[df_p_s_prvs_v.val_new.idxmax()]]['TAG_i-1'].values[0]

        # trace back to t*
        decoded_tags = pd.DataFrame(columns=['SEN_NUM', 'WORD_NUM', 'TAG'])
        counter = num_words - 1
        cur_tag = b_end
        while counter > 0:
            prev_tag = \
                df_b['prev_state'][
                    np.where((df_b['step_idx'] == counter) & (df_b['state'] == cur_tag), 1, 0) == 1].values[
                    0]
            decoded_tags = decoded_tags.append({'SEN_NUM': 0, 'WORD_NUM': counter, 'TAG': cur_tag}, ignore_index=True)
            counter -= 1
            cur_tag = prev_tag

        return decoded_tags

    def is_word_known(self, word):
        if len(self.train_data[np.where(self.train_data['SEG'] == word, 1, 0) == 1]) == 0:
            return False
        else:
            return True

    def evaluate(self, gold_file, test_file, train_file):
        if self.train_data is None:
            self.train(train_file)

        gold_df = ld.load_gold_train(gold_file)
        test_tagged_df = self.decode(test_file)
        eval.output_eval('evaluation/hmm_tagger_eval.txt', model_name="HMM", test_file=test_file, gold_file=gold_file,
                         gold_df=gold_df, test_tagged_df=test_tagged_df)

    def evaluate(self, gold_file, test_file):
        gold_df = ld.load_gold_train(gold_file)
        test_tagged_df = ld.load_gold_train(test_file)
        output_path = os.getcwd() + '\\hmm_tagger.eval'

        eval.output_eval(output_path, model_name="bi-gram/hmm", test_file=test_file,
                         gold_file=gold_file,
                         gold_df=gold_df, test_tagged_df=test_tagged_df)
        print "output basic_tagger.eval file at: {}".format(output_path)

    def output_lex(self):
        rows_segments, columns_tags, lex = self.create_lexical_format()
        output_path = os.getcwd() + '\\hmm_train.lex'
        df = pd.DataFrame(lex)
        df.to_csv(output_path, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\', header=False)
        # for reading: df_test = pd.read_csv(output_path, sep='\t', escapechar='\\')
        print "output hmm_train.lex file at: {}".format(output_path)
        return

    def output_gram(self):
        output_path = os.getcwd() + '\\hmm_train.gram'
        unigram_prob = self.get_unigram_prob()
        unigram_cnt = len(unigram_prob)

        bigram_probs = self.transition_probs[['PROB', 'TAG_i-1', 'TAG_i']]
        bigram_cnt = len(bigram_probs)

        f = open(output_path, "wb")
        f.write("\\data\\" + "\n")
        f.write("ngram 1 ={unigram_cnt}\n".format(unigram_cnt=unigram_cnt))
        f.write("ngram 2 ={bigram_cnt}\n".format(bigram_cnt=bigram_cnt))
        f.write("\n")

        f.write("\\1-grams\\" + "\n")
        unigram_prob.to_csv(f, mode='a', sep='\t', header=False, index=False)
        f.write("\n")

        f.write("\\2-grams\\" + "\n")
        bigram_probs.to_csv(f, mode='a', sep='\t', header=False, index=False)
        f.write("\n")
        f.close()

        print "output hmm_train.gram file at: {}".format(output_path)
        return

    def get_unigram_prob(self):
        unigram_prob = self.train_data[['TAG']]
        unigram_prob['CNT'] = 0
        unigram_prob = unigram_prob.groupby(['TAG'], as_index=False)['CNT'].count()
        unigram_prob['PROB'] = unigram_prob['CNT'] / len(self.train_data)  # TODO- change to log prob
        return unigram_prob[['PROB', 'TAG']]

    def load_lex_from_file(self, lex_file):
        df = pd.read_csv(lex_file, sep='\t', escapechar='\\')
        self.lexical = pd.DataFrame(columns=[['SEG', 'TAG', 'SEG_PROB']])
        for ix, row in df.iterrows():
            seg = row.values[0]
            probs_arr = row.values[1:]
            for tag_prob in probs_arr:
                tag, prob = tag_prob.split('\t')
                if float(prob) > 0:
                    self.lexical = self.lexical.append({'SEG': seg, 'TAG': tag, 'SEG_PROB': float(prob)}, ignore_index=True)
        return

    def load_gram_from_file(self, gram_file):
        with open(gram_file, 'rb') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].find("ngram 1") == 0:
                unigram_rows = int(lines[i].split("=")[1].replace("\n", ""))
            if lines[i].find("ngram 2") == 0:
                bigram_rows = int(lines[i].split("=")[1].replace("\n", ""))

            if lines[i].find("\\1-grams\\") == 0 and unigram_rows is not None:
                unigrams = lines[i + 1:(i + unigram_rows + 1)]
                i = i + unigram_rows
            if lines[i].find("\\2-grams\\") == 0 and bigram_rows is not None:
                bigrams = lines[i+1:(i + bigram_rows+1)]
                i = i + bigram_rows
            i+=1

        self.transition_probs=pd.DataFrame(columns=[['TAG_i', 'TAG_i-1', 'PROB']])
        for transition in bigrams:
            prob,prev_tag,tag=transition.replace('\n','').split('\t')
            prob=float(prob)
            self.transition_probs=self.transition_probs.append({'TAG_i': tag, 'TAG_i-1': prev_tag, 'PROB': prob}, ignore_index=True)
        return

    def output_tagging(self, decoded_df):
        tagged_file_path = os.getcwd() + '\\hmm_tagger.tagged'
        num_sentences=decoded_df.SEN_NUM.max()

        for i in xrange(1,num_sentences+1):
            df=decoded_df[np.where(decoded_df['SEN_NUM']==i,1,0)==1][['SEG','AUTO_TAG']]
            df=df.append({'SEG':'','AUTO_TAG':''},ignore_index=True)
            df.to_csv(tagged_file_path, mode='a', sep='\t',header=False, index=False)

        print "output hmm_tagger.tagged file at: {}".format(tagged_file_path)
        return
        pass


