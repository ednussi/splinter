import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time
import subprocess
import sys
from typing import List
import pdb #TODO:REMOVE


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import spacy
# python3 -m spacy download en_core_web_sm

from copy import deepcopy


def input_data_to_df(input_data):
    df = pd.DataFrame()

    for line in input_data:
        df = df.append(line, ignore_index=True)

    assert len(input_data) == len(df)

    return df

def train_file_to_df(train_file_path):
    # open file
    with open(train_file_path, "r", encoding="utf-8") as reader:
        input_data = [json.loads(line) for line in reader]

    return input_data_to_df(input_data[1:])

def split_qas_to_single_qac_triplets(df):
    split_df = pd.DataFrame()
    for i, row in df.iterrows():
        qas_len = len(row['qas'])
        if qas_len > 1:
            for qas_triplet in row['qas']:
                row_copy = pd.Series(deepcopy(row.to_dict()))
                row_copy['qas'] = [qas_triplet]
                split_df = split_df.append(row_copy, ignore_index=True)
        else:
            split_df = split_df.append(row)

    if len(split_df) != len(df):
        print(f'Splitting original QAS to single single QAC triplets\nStrated with {len(df)} and ended with {len(split_df)}')

    # sanity check
    if not all([len(x)==1 for x in split_df['qas']]):
        exit('problem splitting df')
    
    
    return split_df


def get_words_start_pos_list_spacy(text, tokenizer):
    # There is mismatch with Spacy current parsing of how to treat "'" vs "''"
    tokens = tokenizer(text)
    text_tokens_pos = [[token.text, token.idx] for token in tokens]
    return text_tokens_pos

    #nlp = English()
    #spacy_english_tokenizer = nlp.tokenizer
    # get_words_start_pos_list_spacy(combined_context, spacy_english_tokenizer)

def get_combined_de(de1, de2):
    """

    :param de1:
    :param de2:
    :return: concat context tokens and context in order (de1+de2) and adjust answer's context and context_tokens char_spans
    """

    combined_context = de1['context'] + ' ' + de2['context']
    row2_updated_context_tokens = [[x[0], x[1] + len(de1['context']) + 1] for x in de2['context_tokens']]
    context_tokens = de1['context_tokens'] + row2_updated_context_tokens
    try: # not all mrqa examples have id
        combined_id = de1['id'] + '_' + de2['id']
    except:
        combined_id = ''
    if de2['qas'] == []: #2nd example has no qas
        combined_qas = [de1['qas'][0]]
    else:
        row2_updated_qas = de2['qas'][0] # nothing []
        for det_ans_i, det_ans in enumerate(row2_updated_qas['detected_answers']):
            begin_c = row2_updated_qas['detected_answers'][det_ans_i]['char_spans'][0][0]
            end_c = row2_updated_qas['detected_answers'][det_ans_i]['char_spans'][0][1]
            old_char_span = de2['context'][begin_c:end_c]

            row1_context_length = len(de1['context'])
            char_span_length = row2_updated_qas['detected_answers'][det_ans_i]['char_spans'][0][1] - row2_updated_qas['detected_answers'][det_ans_i]['char_spans'][0][1]
            row2_updated_qas['detected_answers'][det_ans_i]['char_spans'] = [
                [x[0] + row1_context_length + 1, x[1] + row1_context_length + 1] for x in
                row2_updated_qas['detected_answers'][det_ans_i]['char_spans']]
            row2_updated_qas['detected_answers'][det_ans_i]['token_spans'] = [
                [x[0] + len(de1['context_tokens']), x[1] + len(de1['context_tokens'])] for x in
                row2_updated_qas['detected_answers'][det_ans_i]['token_spans']]

            begin_c = row2_updated_qas['detected_answers'][det_ans_i]['char_spans'][0][0]
            end_c = row2_updated_qas['detected_answers'][det_ans_i]['char_spans'][0][1]

        DEBUG_MRQA = False
        if DEBUG_MRQA:
            if combined_context[begin_c:end_c] != de2['qas'][0]['answers'][0].lower() != old_char_span:
                print('Problems:\n')
                time.sleep(1)
                print('new_char_span:', combined_context[begin_c:end_c+1])
                print('Answer in dataexample:', de2['qas'][0]['answers'][0].lower())
                print('old_char_span:', old_char_span)
                pdb.set_trace()
                print('next..')

        if de1['qas'] == []: #in case first example doesn't have a qas
            combined_qas = [row2_updated_qas]
        else:
            combined_qas = [de1['qas'][0], row2_updated_qas] #in case both examples had qas

    return combined_qas, context_tokens, combined_context, combined_id

def qas_pairs_unite(df):
    united_df = pd.DataFrame()
    for i in range(0, len(df), 2):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]
        row1_copy = pd.Series(deepcopy(row1.to_dict()))
        row2_copy = pd.Series(deepcopy(row2.to_dict()))
        # insert in both regular order and oppisite in concatination
        combined_qas, context_tokens, combined_context, combined_id = get_combined_de(row1, row2)
        united_df = united_df.append({'id':combined_id,
                                      'context':combined_context,
                                      'context_tokens':context_tokens,
                                      'qas':combined_qas},ignore_index=True)
        combined_qas, context_tokens, combined_context, combined_id = get_combined_de(row2_copy, row1_copy)
        united_df = united_df.append({'id':combined_id,
                                      'context':combined_context,
                                      'context_tokens':context_tokens,
                                      'qas':combined_qas},ignore_index=True)
    return united_df

def qas_clique_unite(df, seed=None):
    united_df = pd.DataFrame()

    # shuffle df before pairing
    if seed: # optional seed to generate different matching
        np.random.seed(seed)
    df.sample(frac=1) # Actively shuffles

    for i in range(0, len(df)):
        row1 = df.iloc[i]
        for j in range(i+1, len(df)):
            row2 = df.iloc[j]
            row1_copy = pd.Series(deepcopy(row1.to_dict()))
            row2_copy = pd.Series(deepcopy(row2.to_dict()))
            # insert in both regular order and oppisite in concatination
            combined_qas, context_tokens, combined_context, combined_id = get_combined_de(row1, row2)
            united_df = united_df.append({'id':combined_id,
                                          'context':combined_context,
                                          'context_tokens':context_tokens,
                                          'qas':combined_qas},ignore_index=True)
            combined_qas2, context_tokens2, combined_context2, combined_id2 = get_combined_de(row2_copy, row1_copy)
            united_df = united_df.append({'id':combined_id2,
                                          'context':combined_context2,
                                          'context_tokens':context_tokens2,
                                          'qas':combined_qas2},ignore_index=True)
    return united_df

def split_dataframe(df, chunk_size = 8):
    chunk_size = int(chunk_size)
    chunks = list()
    num_chunks = len(df) // int(chunk_size) + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def text_to_MRQA_tokens(text, nlp):
    doc = nlp(text)
    words = [token.text for token in doc]
    spaces = [True if tok.whitespace_ else False for tok in doc]
    # doc2 = spacy.tokens.doc.Doc(nlp.vocab, words=words, spaces=spaces)
    tokens_length = [len(t) for t in words]
    tokens_beginning_chars = [0] + list(np.cumsum(tokens_length[:-1]) + np.cumsum(spaces)[:-1])
    MRQA_text_tokens= [[token_name, token_begining_char] for token_name, token_begining_char in zip(words, tokens_beginning_chars)]

    # Get per sentance
    MRQA_text_tokens_per_sentence = []
    sentences = [sent.text.strip() for sent in doc.sents]
    for sent in sentences:
        # TODO: sent is not used and run for doc every sentance
        words = [token.text for token in doc]
        spaces = [True if tok.whitespace_ else False for tok in doc]
        tokens_length = [len(t) for t in words]
        tokens_beginning_chars = [0] + list(np.cumsum(tokens_length[:-1]) + np.cumsum(spaces)[:-1])
        MRQA_text_tokens_per_sentence.append([[token_name, token_begining_char] for token_name, token_begining_char in zip(words, tokens_beginning_chars)])


    return MRQA_text_tokens, MRQA_text_tokens_per_sentence, words, spaces

def build_MRQA_example(id: str,
                       context: str,
                       questions_id: List[str],
                       questions_qids: List[str],
                       questions_list: List[str],
                       answers_list: List[str],
                       detected_answers_list:  List[List[str]],
                       nlp):
    """

    :param id:
    :param context:
    :param questions_id:
    :param questions_qids:
    :param questions_list:
    :param answers_list:
    :param detected_answers_list:
    :param nlp:
    :return: Feed in simple text and get back an MRQA example in dict form
    """

    context_tokens, MRQA_text_tokens_per_sentence, words, spaces = text_to_MRQA_tokens(context, nlp)

    qas = []
    for q_id, q_qids, q, answers, det_ans in zip(questions_id, questions_qids, questions_list, answers_list, detected_answers_list):
        single_qas = {}
        single_qas['id'] = q_id
        single_qas['qid'] = q_qids
        single_qas['question'] = q
        single_qas['question_tokens'] = text_to_MRQA_tokens(q, nlp) #Weird?
        single_qas['answers'] = answers

        detected_answers = []
        for det_answer in det_ans:
            ans = det_answer['text']

            single_det_ans = {}
            single_det_ans['text'] = ans

            # Inclusive char span
            pdb.set_trace()
            answer_begin_char = context.find(ans)
            answer_end_char = answer_begin_char+len(ans)
            single_det_ans['char_spans'] = [[answer_begin_char, answer_end_char - 1]]

            # Inclusive token span
            ans_doc = nlp(ans)
            ans_words_length = len([token.text for token in ans_doc])
            ans_spaces = [True if tok.whitespace_ else False for tok in ans_doc]

            for i,cont_token in enumerate(context_tokens):
                ans_words = [x[0] for x in context_tokens[i:i+ans_words_length]]
                try:
                    # Will succeed only on things that can combine to an answer
                    recon_ans = str(spacy.tokens.doc.Doc(nlp.vocab, words=ans_words, spaces=ans_spaces))
                except:
                    continue
                if recon_ans == ans:
                    ans_begin_token_ind = i
                    ans_end_token_ind = i + ans_words_length - 1 # Inclusive token span
                    break

            single_det_ans['token_spans'] = [[ans_begin_token_ind, ans_end_token_ind]]

            detected_answers.append(single_det_ans)

            # E.g. [{'text': '$31.5 billion', 'char_spans': [[102, 114]], 'token_spans': [[18, 20]]}]
            single_qas['detected_answers'] = detected_answers

        qas.append(single_qas)

    return {'id': id,
            'context': context,
            'context_tokens': context_tokens,
            'qas': qas}

def sanity_checks_shuffled_vs_regular_row(row, shuffled_row):
    # same context different order - split by space and sort be words
    print_row_example(row)
    print_row_example(shuffled_row)

    assert sorted(row['context'].split(' ')) == sorted(shuffled_row['context'].split(' '))
    for qas, shuffled_qas in zip(row['qas'], shuffled_row['qas']):
        pdb.set_trace()
        assert qas['answers'] == shuffled_qas['answers'], print(qas['answers'], shuffled_qas['answers'])
        assert qas['question'] == shuffled_qas['question'], print(qas['question'], shuffled_qas['question'])
        for det_ans, shuffled_det_ans in zip(qas['detected_answers'], shuffled_qas['detected_answers']):
            print('det_ans', det_ans)
            print('\n')
            print('det_ans', shuffled_det_ans)

            assert det_ans['text'] == shuffled_det_ans['text']

            # Same answer is correctly identified in context text
            row_answer_from_char_span = row['context_tokens'][int(det_ans['char_spans'][0][0]):int(det_ans['char_spans'][0][1])]
            shuffled_answer_from_char_span = shuffled_row['context_tokens'][int(shuffled_det_ans['char_spans'][0][0]):int(shuffled_det_ans['char_spans'][0][1])]
            assert row_answer_from_char_span == shuffled_answer_from_char_span, print(row_answer_from_char_span, shuffled_answer_from_char_span)

            # Same tokens are correctly identified in context tokens
            row_answer_tokens_from_char_span = row['context_tokens'][int(det_ans['token_spans'][0][0]):int(det_ans['token_spans'][0][1])]
            shuffled_answer_tokens_from_char_span = shuffled_row['context_tokens'][int(shuffled_det_ans['token_spans'][0][0]):int(shuffled_det_ans['token_spans'][0][1])]
            assert row_answer_tokens_from_char_span == shuffled_answer_tokens_from_char_span, print(row_answer_tokens_from_char_span,shuffled_answer_tokens_from_char_span)


def rebuild_example_test(row, nlp):
    new_row = build_MRQA_example(id = row['id'],
                                context = row['context'],
                                questions_id = [x['id'] for x in row['qas']],
                                questions_qids = [x['qid'] for x in row['qas']],
                                questions_list = [x['question'] for x in row['qas']],
                                answers_list = [x['answers'] for x in row['qas']],
                                detected_answers_list=[x['detected_answers'] for x in row['qas']],
                                nlp = nlp)

    sanity_checks_shuffled_vs_regular_row(row, new_row)


def shuffle_single_example(row, nlp):

    # when shuffling context we need to update:

    # Shuffle
    context = row['context']
    context_tokens = row['context_tokens']
    doc = nlp(context)
    # for token in doc:
    #     print(token.text, token.pos_, token.dep_)
    sentences = [sent.text.strip() for sent in doc.sents]
    orig_sent_breakdown = [sent.text.strip() for sent in doc.sents]
    sentences_length = [len(sent) for sent in sentences]
    # such that context[index] is the actual first character of sentance
    sentences_begin_inds = np.array([0] + list(np.cumsum(sentences_length[:-1]) + np.array(range(1, len(sentences_length)))))

    # If answer spans over multiple sentances, combine them
    for row_qas in row['qas']:
        for det_answer in row_qas['detected_answers']:
            # Get Answer
            answer_start_ind = int(det_answer['char_spans'][0][0])
            answer_len = int(det_answer['char_spans'][0][1]) - answer_start_ind
            answer_end_ind = answer_start_ind + answer_len
            # find sentences ind answer is contained in
            answer_start_sent_ind =  sum(answer_start_ind >= sentences_begin_inds) - 1
            answer_end_sent_ind = sum(answer_end_ind >= sentences_begin_inds) - 1
            # combines these sentences
            if answer_start_sent_ind != answer_end_sent_ind:
                sentences = sentences[:answer_start_sent_ind] + [' '.join(sentences[answer_start_sent_ind:answer_end_sent_ind+1])] + sentences[answer_end_sent_ind+1:]
                sentences_length = [len(sent) for sent in sentences]
                # such that context[index] is the actual first character of sentance
                sentences_begin_inds = np.array(
                    [0] + list(np.cumsum(sentences_length[:-1]) + np.array(range(1, len(sentences_length)))))
    # Shuffle Context and Context Tokens
    new_sent_order_inds = np.random.permutation(len(sentences))
    shuffled_sentences = [sentences[sent_order] for sent_order in new_sent_order_inds]
    shuffled_sentences_length = [len(sent) for sent in shuffled_sentences]
    shuffled_sentences_begin_inds = np.array([0] + list(np.cumsum(shuffled_sentences_length[:-1]) + np.array(range(len(shuffled_sentences_length) - 1))))
    shuffled_context = ' '.join(shuffled_sentences) #Changing
    shuffled_tokens, shuffled_MRQA_text_tokens_per_sentence, shuffled_context_words, shuffled_context_spaces = text_to_MRQA_tokens(shuffled_context, nlp) #Changing

    qas = []
    for row_qas in row['qas']:
        single_qas = {}
        single_qas['id'] = row_qas['id']
        single_qas['qid'] = row_qas['qid']
        single_qas['question'] = row_qas['question']
        single_qas['question_tokens'] = row_qas['question_tokens']
        single_qas['answers'] = row_qas['answers']

        detected_answers = []
        DEBUG = False
        for det_answer in row_qas['detected_answers']:
            single_det_ans = {}
            single_det_ans['text'] = det_answer['text']
            answer_start_ind = int(det_answer['char_spans'][0][0])
            answer_len = int(det_answer['char_spans'][0][1]) - answer_start_ind
            answer_end_ind = answer_start_ind + answer_len
            old_ans = context[answer_start_ind:answer_end_ind + 1]
            old_tokens = context_tokens[int(det_answer['token_spans'][0][0]):int(det_answer['token_spans'][0][1])+1]
            sentence_answer_ind_in_shuffled_sents = len(shuffled_sentences_begin_inds) - 1 #last sentance index
            sentence_answer_ind_in_orig_sents = len(sentences_begin_inds) - 1 #last sentance index
            
            # Find to what sentence original answer maps into original context
            sentence_answer_ind_in_orig_sents = sum(answer_start_ind >= sentences_begin_inds) - 1
            answer_char_offset_from_sentance_answer = answer_start_ind - sentences_begin_inds[sentence_answer_ind_in_orig_sents]
            # Test offset is correct
            if not sentences[sentence_answer_ind_in_orig_sents][answer_char_offset_from_sentance_answer:].startswith(old_ans):
                if sentences[sentence_answer_ind_in_orig_sents][answer_char_offset_from_sentance_answer-1:].startswith(old_ans):
                    answer_char_offset_from_sentance_answer = answer_char_offset_from_sentance_answer - 1 #TODO Why? A few answer have the original indices not in place
                else:
                    print('WARNING: Tokens Mismatch')
                    print('old_ans', old_ans)
                    print('sentences[sentence_answer_ind_in_orig_sents][answer_char_offset_from_sentance_answer:]', sentences[sentence_answer_ind_in_orig_sents][answer_char_offset_from_sentance_answer:])
                    # pdb.set_trace()

            # Find to what sentence original answers maps into
            sentence_answer_ind_in_shuffled_sents = np.where(new_sent_order_inds == sentence_answer_ind_in_orig_sents)[0][0] 
            if old_ans not in shuffled_sentences[sentence_answer_ind_in_shuffled_sents]:
                print('old_ans', old_ans)
                print( 'shuffled_sentences[sentence_answer_ind_in_shuffled_sents]', shuffled_sentences[sentence_answer_ind_in_shuffled_sents])
                pdb.set_trace()
                print('Bad \'sentence_answer_ind_in_shuffled_sents\' index')

            # # Find how many chars from beginning of sentence is answer
            # new_ans_sent = shuffled_sentences[sentence_answer_ind_in_shuffled_sents][answer_char_offset_from_sentance_answer:]
            # if DEBUG:
            #     print('answer_char_offset_from_sentance_answer', answer_char_offset_from_sentance_answer)
            #     print('new_ans_sent', new_ans_sent)

            # Find how many character lead to where sentenace is now positioned
            new_index_of_shuffled_sentance = np.where(new_sent_order_inds == sentence_answer_ind_in_orig_sents)[0][0] #Only one match
            text_before_shuffled_answer = ' '.join(shuffled_sentences[:new_index_of_shuffled_sentance])
            # sanity
            is_first_sentance = bool(new_index_of_shuffled_sentance)
            shuffled_answer_start_ind = len(text_before_shuffled_answer) + answer_char_offset_from_sentance_answer + int(is_first_sentance)
            shuffled_answer_end_ind = shuffled_answer_start_ind + answer_len
            single_det_ans['char_spans'] = [[shuffled_answer_start_ind, shuffled_answer_end_ind]]  # Changing

            # Verify Same answer is correctly identified in context - Add one to last index because indices are inclusive
            new_ans = shuffled_context[shuffled_answer_start_ind:shuffled_answer_end_ind+1]
            if old_ans != new_ans:
                print('WARNING: Answers Mismatch')
                print('old_ans', old_ans)
                print('new_ans', new_ans)
                # pdb.set_trace()
                # assert old_ans == new_ans, print('old_ans == new_ans',old_ans,new_ans)

            shuffled_answer_start_ind, shuffled_answer_end_ind
            shuffled_answer_token_start_ind = sum([shuffled_answer_start_ind > x[1] for x in shuffled_tokens])
            shuffled_answer_token_end_ind = sum([shuffled_answer_end_ind >= x[1] for x in shuffled_tokens]) - 1 #Inclusive so removing 1 for span
            single_det_ans['token_spans'] = [[shuffled_answer_token_start_ind, shuffled_answer_token_end_ind]] #Changing

            # Sanity checks - tokens spans create same answer
            if not [x[0] for x in shuffled_tokens[shuffled_answer_token_start_ind:shuffled_answer_token_end_ind+1]] == [x[0] for x in old_tokens]:
                print('WARINING: Tokens Mismatch')
                print('original ans tokens', [x[0] for x in old_tokens])
                print('shuffled ans tokens', [x[0] for x in shuffled_tokens[shuffled_answer_token_start_ind:shuffled_answer_token_end_ind+1]])
                """
                shuffled ans tokens ['The', 'lights', 'can', 'be', 'switched', 'on', 'for', '24', '-', 'hrs', '/', 'day', ',', 'or', 'a', 'range', 'of', 'step', '-', 'wise', 'light', 'regimens', 'to', 'encourage', 'the', 'birds', 'to', 'feed', 'often', 'and', 'therefore', 'grow', 'rapidly']
                original ans tokens ['The', 'lights', 'can', 'be', 'switched', 'on', 'for', '24-hrs', '/', 'day', ',', 'or', 'a', 'range', 'of', 'step', '-', 'wise', 'light', 'regimens', 'to', 'encourage', 'the', 'birds', 'to', 'feed', 'often', 'and', 'therefore', 'grow', 'rapidly']
                """
                # pdb.set_trace()
                
                # assert [x[0] for x in shuffled_tokens[shuffled_answer_token_start_ind:shuffled_answer_token_end_ind]] == [x[0] for x in original_tokens]
            detected_answers.append(single_det_ans)

        single_qas['detected_answers'] = detected_answers

        qas.append(single_qas)

    return {'id': row['id'],
            'context': shuffled_context,
            'context_tokens': shuffled_tokens,
            'qas': qas}


def get_nlp():
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    return nlp

def shuffle_context(df, seed=None):
    nlp = get_nlp()

    # set numpy seed here to get perfect results every time
    shuffled_df = pd.DataFrame()
    for i in tqdm(range(0, len(df)), desc='Creating Shuffle Augs'):
        row_copy = pd.Series(deepcopy(df.iloc[i].to_dict()))
        single_qas = shuffle_single_example(row_copy, nlp)
        shuffled_df = shuffled_df.append(single_qas, ignore_index=True)
    return shuffled_df


def concat_single_example(row, text_to_concat, before=True):
    nlp = get_nlp()

    text_to_concat_tokens, _, _, _ = text_to_MRQA_tokens(text_to_concat, nlp)
    de2 = {'id': 'lor_ip',
           'context': text_to_concat,
           'context_tokens': text_to_concat_tokens,
           'qas': []}
    # get_combined_de does changes to the original struct passed
    row_copy = pd.Series(deepcopy(row.to_dict()))
    de2_copy = pd.Series(deepcopy(de2))
    if before:
        de2['id'] += '_b'
        qas, concat_context_tokens, concat_context, id = get_combined_de(de2, row_copy)
    else:
        de2['id'] += '_a'
        qas, concat_context_tokens, concat_context, id = get_combined_de(row, de2_copy)

    return {'id': id,
            'context': concat_context,
            'context_tokens': concat_context_tokens,
            'qas': qas}

def concat_text(df, texts_to_concat, both, max_seq_length):
    # set numpy seed here to get perfect results every time
    concat_text_df = pd.DataFrame()
    for i in tqdm(range(0, len(df)), desc='Creating Concat Augs'):
        row_copy = pd.Series(deepcopy(df.iloc[i].to_dict()))

        # When combining need to make sure the signal isn't lost as max_seq_length cuts tokens
        # need to shorten the text_to_concat to match the max_seq_length
        # such that the answer and signal from orig text won't disappear
        text_to_concat = texts_to_concat[i]
        max_other_tokens = max_seq_length - len(row_copy['context_tokens'])
        max_other_words = round(max_other_tokens * 0.75) # using a fix 1 word ~= 4 tokens approximated ratio
        text_to_concat = " ".join(text_to_concat.split()[:max_other_words])
        if not text_to_concat.endswith('.'):
            text_to_concat += '.'

        if both:
            single_qas_before = concat_single_example(row_copy, text_to_concat, before=True)
            single_qas_after = concat_single_example(row_copy, text_to_concat, before=False)
            # print_row_example(single_qas_before)
            # pdb.set_trace()
            # pdb.set_trace()
            # print_row_example(single_qas_after)
            # pdb.set_trace()
            concat_text_df = concat_text_df.append(single_qas_before, ignore_index=True)
            concat_text_df = concat_text_df.append(single_qas_after, ignore_index=True)
        else:
            concat_before = bool(round(np.random.rand())) #randomly roll where to concat text
            single_qas = concat_single_example(row_copy, text_to_concat, before=concat_before)
            # print(concat_before)
            # print_row_example(single_qas)
            # pdb.set_trace()
            concat_text_df = concat_text_df.append(single_qas, ignore_index=True)
    return concat_text_df

def qas_npairs_unite(df, pairs, seed=None):
    # Divide and runs in clique - aggrigate in end
    npairs_df = pd.DataFrame()
    for df_chunk in split_dataframe(df, chunk_size=pairs):
        united_df = qas_clique_unite(df_chunk, seed)
        npairs_df = pd.concat([npairs_df, united_df], ignore_index=True)

    return npairs_df

def write_df(df, name):
    new_jsonl_lines = []
    header = {"header": {"dataset": "SQuAD", "split": "train"}}

    for i,row in df.iterrows():
        new_jsonl_lines.append(row.to_dict())

    # Write as a new jsonl file
    print(f'Writing new augmented data file to {name}')
    with open(name, "w", encoding="utf-8") as writer:
        writer.write(f'{json.dumps(header)}\n')
        for line in new_jsonl_lines:
            writer.write(f'{json.dumps(line)}\n')
    return


def create_moasic_unite_exp_data(squad_path):
    exp_name = 'mosaic_unite'
    # open folder for expirement
    output_dir = f'{squad_path}/{exp_name}'
    while os.path.exists(output_dir):
        exp_name += '_new'
        output_dir = f'{squad_path}/{exp_name}'
    os.mkdir(output_dir)
    for seed in tqdm([42, 43, 44, 45, 46], desc='Seeds'):
        for num_examples in tqdm([16, 32, 64, 128, 256], desc='Examples Num'):
            train_file_name = f'squad-train-seed-{seed}-num-examples-{num_examples}.jsonl'
            df = train_file_to_df(f'{squad_path}/{train_file_name}')
            df = split_qas_to_single_qac_triplets(df)
            uni_df = qas_pairs_unite(df)
            write_df(uni_df, f'{output_dir}/squad-train-seed-{seed}-num-examples-{num_examples}.jsonl')

def create_moasic_unite_npairs_exp_data(squad_path, pairs=8, final_single_qac_triplets=False):
    exp_name = f'mosaic_unite_npairs-{pairs}'
    if final_single_qac_triplets:
        exp_name+='-singleqac'
    # open folder for expirement
    output_dir = f'{squad_path}/{exp_name}'
    while os.path.exists(output_dir):
        exp_name += '_new'
        output_dir = f'{squad_path}/{exp_name}'
    os.mkdir(output_dir)
    for num_examples in tqdm([256, 128, 64, 32, 16], desc='Examples Num'):
        for seed in tqdm([42, 43, 44, 45, 46], desc='Seeds'):
            train_file_name = f'baseline/squad-train-seed-{seed}-num-examples-{num_examples}.jsonl'
            df = train_file_to_df(f'{squad_path}/{train_file_name}')
            df = split_qas_to_single_qac_triplets(df)
            uni_df = qas_npairs_unite(df, pairs)

            if final_single_qac_triplets:
                uni_df = split_qas_to_single_qac_triplets(uni_df)
            write_df(uni_df, f'{output_dir}/squad-train-seed-{seed}-num-examples-{num_examples}.jsonl')


def print_row_example(row):
    print("="*15 + row['id'] + "="*15)
    print('\n'+"="*15 + 'QAS' + "="*15)
    for qas in row['qas']:
        ans = qas['answers']
        q = qas['question']
        print(f"Question: {q}")
        print(f"Answer: {ans}")
        for det_answer in qas['detected_answers']:
            # Get Answer
            answer_start_ind = int(det_answer['char_spans'][0][0])
            answer_len = int(det_answer['char_spans'][0][1]) - answer_start_ind
            answer_end_ind = answer_start_ind + answer_len
            ans_from_char_spans = row['context'][answer_start_ind:answer_end_ind+1]
        print(f"Answer from char_spans: {ans_from_char_spans}")
    print('\n'+"="*15 + 'Context' + "="*15)
    print(row['context'])
    # print('\n'+"="*15 + 'Context Tokens' + "="*15)
    # print(row['context_tokens'])

def print_df_example(df, index=0):
    row = df.iloc[index]
    print_row_example(row)


def crop_single_qas(row):
    nlp = get_nlp()

    # Find where the context is
    context = row['context']
    context_tokens = row['context_tokens']
    doc = nlp(context)
    # for token in doc:
    #     print(token.text, token.pos_, token.dep_)
    sentences = [sent.text.strip() for sent in doc.sents]
    orig_sent_breakdown = [sent.text.strip() for sent in doc.sents]
    sentences_length = [len(sent) for sent in sentences]
    # such that context[index] is the actual first character of sentance
    sentences_begin_inds = np.array([0] + list(np.cumsum(sentences_length[:-1]) + np.array(range(1, len(sentences_length)))))

    # If answer spans over multiple sentances, combine them
    for row_qas in row['qas']:
        for det_answer in row_qas['detected_answers']:
            # Get Answer
            answer_start_ind = int(det_answer['char_spans'][0][0])
            answer_len = int(det_answer['char_spans'][0][1]) - answer_start_ind
            answer_end_ind = answer_start_ind + answer_len
            # find sentences ind answer is contained in
            answer_start_sent_ind =  sum(answer_start_ind >= sentences_begin_inds) - 1
            answer_end_sent_ind = sum(answer_end_ind >= sentences_begin_inds) - 1
            # combines these sentences
            if answer_start_sent_ind != answer_end_sent_ind:
                sentences = sentences[:answer_start_sent_ind] + [' '.join(sentences[answer_start_sent_ind:answer_end_sent_ind+1])] + sentences[answer_end_sent_ind+1:]
                sentences_length = [len(sent) for sent in sentences]
                # such that context[index] is the actual first character of sentance
                sentences_begin_inds = np.array(
                    [0] + list(np.cumsum(sentences_length[:-1]) + np.array(range(1, len(sentences_length)))))

    # CROP - Drop at random sentances, before / after signal(answer)
    # Drop after
    end_crop_index = len(sentences)
    if answer_start_sent_ind < len(sentences):
        end_crop_index = np.random.randint(answer_start_sent_ind + 1, len(sentences)+1)

    # Drop before
    start_crop_index = 0
    if answer_start_sent_ind >= 1:
        start_crop_index = np.random.randint(answer_start_sent_ind + 1)

    # uniform distribution to drop #before, and after# - potentially could keep entire sentence
    cropped_sentences = sentences[start_crop_index:end_crop_index]
    cropped_context = ' '.join(cropped_sentences)
    cropped_tokens, cropped_MRQA_text_tokens_per_sentence, cropped_context_words, cropped_context_spaces = text_to_MRQA_tokens(cropped_context, nlp) #Changing

    # only dropped sentences from b4 effect the answer token/char spans
    chars_dropped = sentences_begin_inds[start_crop_index]
    tokens_till_crop = sum([x[1] < chars_dropped for x in context_tokens])
    num_tokens_dropped = len(context_tokens[:tokens_till_crop])

    qas = []
    for row_qas in row['qas']:
        single_qas = {}
        if 'id' in row_qas.keys():
            new_id = row_qas['id']
        else:
            new_id = ''
        single_qas['id'] = new_id
        single_qas['qid'] = row_qas['qid']
        single_qas['question'] = row_qas['question']
        single_qas['question_tokens'] = row_qas['question_tokens']
        single_qas['answers'] = row_qas['answers']

        detected_answers = []
        for det_answer in row_qas['detected_answers']:
            single_det_ans = {}
            single_det_ans['text'] = det_answer['text']
            answer_start_ind = int(det_answer['char_spans'][0][0])
            answer_len = int(det_answer['char_spans'][0][1]) - answer_start_ind
            answer_end_ind = answer_start_ind + answer_len
            old_ans = context[answer_start_ind:answer_end_ind + 1]
            old_tokens = context_tokens[int(det_answer['token_spans'][0][0]):int(det_answer['token_spans'][0][1])+1]

            # remove the chopped chars in the beginning
            cropped_answer_start_ind = int(det_answer['char_spans'][0][0]) - chars_dropped
            cropped_answer_end_ind = int(det_answer['char_spans'][0][1]) - chars_dropped

            single_det_ans['char_spans'] = [[cropped_answer_start_ind, cropped_answer_end_ind]]
            # TODO: correct single shift since tokenizer are not exactly the same
            if old_ans != cropped_context[cropped_answer_start_ind:cropped_answer_end_ind + 1]:
                fixed = False
                for i in range(1,len(sentences)):

                    if old_ans == cropped_context[cropped_answer_start_ind+i:cropped_answer_end_ind+i + 1]:
                        cropped_answer_start_ind += i
                        cropped_answer_end_ind += i
                        fixed=True
                        break
                    elif old_ans == cropped_context[cropped_answer_start_ind-i:cropped_answer_end_ind -i +1]:
                        cropped_answer_start_ind -= i
                        cropped_answer_end_ind -= i
                        fixed = True
                        break
                if not fixed:
                    print('Couldnt fix ans')
                    continue # lossing the ans


            # Sanity checks
            new_ans = cropped_context[cropped_answer_start_ind:cropped_answer_end_ind + 1]
            if old_ans != new_ans:
                import pdb; pdb.set_trace()
                print('WARNING: Answers Mismatch')
                print('old_ans', old_ans)
                print('new_ans', new_ans)

            # remove the chopped tokens in the beginning
            cropped_answer_token_start_ind = sum([cropped_answer_start_ind > x[1] for x in cropped_tokens])
            cropped_answer_token_end_ind = sum(
                [cropped_answer_end_ind >= x[1] for x in cropped_tokens]) - 1  # Inclusive so removing 1 for span
            single_det_ans['token_spans'] = [
                [cropped_answer_token_start_ind, cropped_answer_token_end_ind]]  # Changing



            if not [x[0] for x in
                    cropped_tokens[cropped_answer_token_start_ind:cropped_answer_token_end_ind + 1]] == \
                   [x[0] for x in old_tokens]:
                print('WARINING: Tokens Mismatch')
                print('original ans tokens', [x[0] for x in old_tokens])
                print('cropped ans tokens', [x[0] for x in cropped_tokens[
                                                            cropped_answer_token_start_ind:cropped_answer_token_end_ind + 1]])

            detected_answers.append(single_det_ans)
        
        single_qas['detected_answers'] = detected_answers
        qas.append(single_qas)

    if 'id' in row_qas.keys():
        new_id = row_qas['id']
    else:
        new_id = ''

    return {'id': new_id,
            'context': cropped_context,
            'context_tokens': cropped_tokens,
            'qas': qas}

def crop_qas_df(df):
    cropped_df = pd.DataFrame()
    for i in tqdm(range(0, len(df)), desc='Creating Concat Augs'):
        row = pd.Series(deepcopy(df.iloc[i].to_dict()))
        row = crop_single_qas(row)
        cropped_df = cropped_df.append(row, ignore_index=True)
    return cropped_df

def mosaic_npairs_single_qac_aug(input_data, pairs=2, final_single_qac_triplets=True, seed=None, crop=False):
    df = input_data_to_df(input_data)
    split_df = split_qas_to_single_qac_triplets(df)
    if crop:
        split_df = crop_qas_df(split_df)
    uni_df = qas_npairs_unite(split_df, pairs, seed)
    if final_single_qac_triplets:
        uni_single_qac_df = split_qas_to_single_qac_triplets(uni_df)
        return uni_single_qac_df
    return uni_df

def context_shuffle_aug(input_data):
    df = input_data_to_df(input_data)
    split_df = split_qas_to_single_qac_triplets(df)
    random_sent_order_df = shuffle_context(split_df)
    return random_sent_order_df

def concat_lorem_ipsum(input_data, both=False, max_seq_length=328):
    LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    df = input_data_to_df(input_data)
    split_df = split_qas_to_single_qac_triplets(df)
    texts_to_concat = [LOREM_IPSUM] * len(split_df)
    return concat_text(split_df, texts_to_concat, both, max_seq_length)

def concat_coherent_text(input_data, max_seq_length):

    df = input_data_to_df(input_data)
    split_df = split_qas_to_single_qac_triplets(df)

    # Get data from a different MRQA data for coherent text
    datasets = ['bioasq', 'hotpotqa', 'naturalquestions', 'newsqa', 'searchqa', 'textbookqa', 'triviaqa']
    # pick a non squad dataset at random
    dataset = datasets[np.random.randint(0, len(datasets)-1)]
    dataset_filepath =f'mrqa_data/{dataset}/{dataset}-train-seed-42-num-examples-256.jsonl'
    with open(dataset_filepath, "r", encoding="utf-8") as reader:
        print(reader.readline())
        dataset_input_data = [json.loads(line) for line in reader]
    df = input_data_to_df(dataset_input_data)
    texts_to_concat = []
    for i in range(len(split_df)):
        row = df.sample()
        # pdb.set_trace()
        text_to_concat = row['context'].values[0]
        text_to_concat = " ".join(text_to_concat.split()) #remove double spaces and multiple \n
        texts_to_concat.append(text_to_concat)

    return concat_text(split_df, texts_to_concat, both=True, max_seq_length=max_seq_length)

def concat_single_coherent_text(input_data):

    # Get data from a different MRQA data for coherent text
    datasets = ['bioasq', 'hotpotqa', 'naturalquestions', 'newsqa', 'searchqa', 'textbookqa', 'triviaqa']
    # pick a non squad dataset at random
    dataset = datasets[np.random.randint(0, len(datasets)-1)]
    dataset_filepath =f'mrqa_data/{dataset}/{dataset}-train-seed-42-num-examples-256.jsonl'
    with open(dataset_filepath, "r", encoding="utf-8") as reader:
        print(reader.readline())
        dataset_input_data = [json.loads(line) for line in reader]
    df = input_data_to_df(dataset_input_data)
    row = df.sample()
    # pdb.set_trace()
    text_to_concat = row['context'].values[0]
    text_to_concat = " ".join(text_to_concat.split()) #remove double spaces and multiple \n
    texts_to_concat = [text_to_concat] * len(input_data)
    return concat_text(input_data, texts_to_concat, both=True)

if __name__ == '__main__':
    # train_file_name = f'squad/moasic_unite/squad-train-seed-42-num-examples-16.jsonl'
    # df = train_file_to_df(train_file_name)
    # df2 = split_qas_to_single_qac_triplets(df)
    # import pdb; pdb.set_trace()
    # print('end')
    # Create Data
    squad_path = 'squad'
    create_moasic_unite_npairs_exp_data(squad_path, pairs=2, final_single_qac_triplets=True)
    create_moasic_unite_npairs_exp_data(squad_path, pairs=4, final_single_qac_triplets=True)
    create_moasic_unite_npairs_exp_data(squad_path, pairs=8, final_single_qac_triplets=True)