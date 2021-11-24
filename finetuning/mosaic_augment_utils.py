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
        print(f'Strated with {len(df)} and ended with {len(split_df)}')

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
    combined_context = de1['context'] + ' ' + de2['context']
    row2_updated_context_tokens = [[x[0], x[1] + len(de1['context']) + 1] for x in de2['context_tokens']]
    context_tokens = de1['context_tokens'] + row2_updated_context_tokens
    combined_id = de1['id'] + '_' + de2['id']
    row2_updated_qas = de2['qas'][0]
    for det_ans_i, det_ans in enumerate(row2_updated_qas['detected_answers']):
        begin_c = row2_updated_qas['detected_answers'][det_ans_i]['char_spans'][0][0]
        end_c = row2_updated_qas['detected_answers'][det_ans_i]['char_spans'][0][1]
        old_char_span = de2['context'][begin_c:end_c+1]

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
                print('Answer in dataexample:', de2['qas'][0]['answers'][0])
                print('old_char_span:', old_char_span)
                pdb.set_trace()
                print('next..')

    combined_qas = [de1['qas'][0], row2_updated_qas]
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
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
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
    detected_answers_inds
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
    sentences_length = [len(sent) for sent in sentences]
    # such that context[index] is the actual first character of sentance
    sentences_begin_inds = np.array([0] + list(np.cumsum(sentences_length[:-1]) + np.array(range(1, len(sentences_length)))))

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
                print('old_ans', old_ans)
                print('sentences[sentence_answer_ind_in_orig_sents][answer_char_offset_from_sentance_answer:]', sentences[sentence_answer_ind_in_orig_sents][answer_char_offset_from_sentance_answer:])
                pdb.set_trace()
                print('DEBUG3')

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
                print('old_ans', old_ans)
                print( 'new_ans', new_ans)
                pdb.set_trace()
                print('DEBUG2!!')

            assert old_ans == new_ans, print('old_ans == new_ans',old_ans,new_ans)

            shuffled_answer_start_ind, shuffled_answer_end_ind
            shuffled_answer_token_start_ind = sum([shuffled_answer_start_ind > x[1] for x in shuffled_tokens])
            shuffled_answer_token_end_ind = sum([shuffled_answer_end_ind > x[1] for x in shuffled_tokens]) - 1 #Inclusive so removing 1 for span
            single_det_ans['token_spans'] = [[shuffled_answer_token_start_ind, shuffled_answer_token_end_ind]] #Changing

            # Sanity checks - tokens spans create same answer
            original_tokens = context_tokens[int(det_answer['token_spans'][0][0]):int(det_answer['token_spans'][0][1])]
            assert [x[0] for x in shuffled_tokens[shuffled_answer_token_start_ind:shuffled_answer_token_end_ind]] == [x[0] for x in original_tokens]
            detected_answers.append(single_det_ans)

        single_qas['detected_answers'] = detected_answers

        qas.append(single_qas)

    return {'id': row['id'],
            'context': shuffled_context,
            'context_tokens': shuffled_tokens,
            'qas': qas}


def shuffle_context(df, seed=None):
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    # set numpy seed here to get perfect results every time
    shuffled_df = pd.DataFrame()
    for i in tqdm(range(0, len(df)), desc='Creating Shuffle Augs'):
        row_copy = pd.Series(deepcopy(df.iloc[i].to_dict()))
        single_qas = shuffle_single_example(row_copy, nlp)
        shuffled_df = shuffled_df.append(single_qas, ignore_index=True)
    return shuffled_df

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
    print('\n'+"="*15 + 'Context' + "="*15)
    print(row['context'])
    # print('\n'+"="*15 + 'Context Tokens' + "="*15)
    # print(row['context_tokens'])

def print_df_example(df, index=0):
    row = df.iloc[index]
    print_row_example(row)

def mosaic_npairs_single_qac_aug(input_data, pairs=2, final_single_qac_triplets=True, seed=None):
    df = input_data_to_df(input_data)
    split_df = split_qas_to_single_qac_triplets(df)
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


def random_text_add_aug(input_data):
    df = input_data_to_df(input_data)
    split_df = split_qas_to_single_qac_triplets(df)
    random_text_added_df = add_random_text(split_df)
    return random_text_added_df   

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