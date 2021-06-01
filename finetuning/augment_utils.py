import os
import json
import gzip
import shutil
import nlpaug.augmenter.word as naw
import requests
import re
import logging
logger = logging.getLogger(__name__)
from utils import init_parser, get_aug_filename
from tqdm import tqdm
from spacy.lang.en import English
import random
import torch
import numpy as np

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def download(url, fname):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def get_augs_from_names(augs_names):
    googlenews_bin = 'data/GoogleNews-vectors-negative300.bin'

    if not os.path.exists(googlenews_bin):
        if not os.path.exists(googlenews_bin.split('/')[0]):
            os.mkdir(googlenews_bin.split('/')[0])

        logger.info('Downloading GoogleNews-vectors-negative300.bin.gz')
        googlenews_bin_gz_url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

        download(googlenews_bin_gz_url, f'{googlenews_bin}.gz')
        logger.info('Unzipping gz file')
        # un-gz file and save as bin
        with gzip.open(f'{googlenews_bin}.gz', 'rb') as f_in:
            with open(googlenews_bin, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info('Done')

    logger.info(f'Loading Augs {augs_names}')
    switcher = {
        'delete-random': naw.RandomWordAug(),
        'insert-word-embed':naw.WordEmbsAug(model_type='word2vec',
                                model_path=googlenews_bin,
                                action="insert"),
        'sub-word-embed': naw.WordEmbsAug(model_type='word2vec',
                                model_path=googlenews_bin,
                                action="substitute"),
        'insert-bert-embed': naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert"),
        'sub-bert-embed': naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
    }

    return [switcher.get(aug_name, None) for aug_name in augs_names]

def get_n_new_aug(aug, text, new_augs_num):
    news_augs_set = {aug.augment(text) for i in range(new_augs_num)}
    while len(news_augs_set) < new_augs_num:
        news_augs_set.update({aug.augment(text)})
    return news_augs_set

def get_words_start_pos_list_spacy(text, tokenizer):
    # There is mismatch with Spacy current parsing of how to treat "'" vs "''"
    tokens = tokenizer(text)
    text_tokens_pos = [[token.text, token.idx] for token in tokens]
    return text_tokens_pos

def get_words_start_pos_list(text, tokenizer):
    encoded_input = tokenizer(text)
    print(encoded_input)
    words_tokenized = [tokenizer.decode(token) for token in encoded_input['input_ids']]
    print(words_tokenized)
    words_tokenized = [w.strip() for w in words_tokenized if w not in ['<s>','</s>']]
    text_tokens = [[word.group(), word.start()] for word in re.finditer(r'\S+', ' '.join(words_tokenized))]
    text_tokens[-1][1] = text_tokens[-1][1] - 1 # question mark indicatng last token?
    print(text_tokens)
    return text_tokens


def add_aug(args, new_f_name):
    logger.info(f'Loading Examples from file {args.train_file}')
    # open file
    with open(args.train_file, "r", encoding="utf-8") as reader:
        input_data = [json.loads(line) for line in reader]

    augs = get_augs_from_names(args.augs_names)
    logger.info(f'Loading Tokenizer {args.tokenizer_name}')
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, add_prefix_space=True) #add_prefix_space=True for roberta-base

    # Get Spacy tokenizer with the default settings for English including punctuation rules and exceptions
    nlp = English()
    tokenizer = nlp.tokenizer

    new_jsonl_lines = []
    header_line = input_data[0]
    for line in tqdm(input_data[1:], desc='Enhancing with augs'):
        line_copy = line.copy()
        new_jsonl_lines.append(line)
        # for every q-a-c example stored
        for qas in line['qas']:
            answers = qas['answers']
            q = qas['question']
            id = qas['id']
            qid = qas['qid']
            # q_tokens = qas['question_tokens']
            detected_answers = qas['detected_answers']

            # Sanity check to make sure question tokens are reconstructed correctly
            q_tokens = get_words_start_pos_list_spacy(q, tokenizer)
            # if  q_tokens != qas['question_tokens']:
            #     import pdb; pdb.set_trace()
            # assert q_tokens == qas['question_tokens'], f'{q_tokens},\n {qas["question_tokens"]} '

            for aug_num,aug in zip(args.augs_count, augs):
                # create augs per count
                set_seed(args) #initialize seed
                news_q_augs_set = get_n_new_aug(aug, q, int(aug_num))

                for new_aug_q_i, new_aug_q in enumerate(news_q_augs_set):
                    # question/answer/context tokens in the jsonl file are just the position each word starts in
                    new_question_tokens = get_words_start_pos_list_spacy(new_aug_q, tokenizer)

                    # # Save in MRQAExample internal format
                    # MRQAExample(qas_id=qas_id, question_text=question_text, question_tokens=question_tokens,
                    #             context_text=context, context_tokens=context_tokens,
                    #             answer_text=answer_text,
                    #             start_position_character=start_position_character, answers=answers)

                    # append new example
                    augmented_qas = {'answers':answers,
                                           'question':new_aug_q,
                                           'id': f'{id}_{new_aug_q_i}',
                                           'qid': f'{qid}_{new_aug_q_i}',
                                           'question_tokens': new_question_tokens,
                                           'detected_answers': detected_answers}

                    line_copy['qas'] = [augmented_qas]
                    # Now add all the new questions to this line
                    new_jsonl_lines.append(line_copy)

    # Write as a new jsonl file
    logger.info(f'Writing new augmented data file to {new_f_name}')
    with open(new_f_name, "w", encoding="utf-8") as writer:
        writer.write(f'{json.dumps(header_line)}\n')
        for line in new_jsonl_lines:
            writer.write(f'{json.dumps(line)}\n')
    return

def get_args(seed=42,
             train_file="squad/squad-train-seed-42-num-examples-16.jsonl",
             augs_names='sub-bert-embed',
             tokenizer_name='roberta-base',
             augs_count='1',
             model_type='roberta-base',
             model_name_or_path='',
             output_dir='',
):

    # Create a Namespace immulating argparser
    class C:
        pass
    args = C()
    parser = init_parser()
    parser.parse_args(args=['--seed', seed,
                            '--train_file', train_file,
                            '--augs_names', augs_names,
                            '--tokenizer_name', tokenizer_name,
                            '--augs_count', augs_count,
                            '--model_type', model_type, # 3 required params
                            '--model_name_or_path', model_name_or_path,
                            '--output_dir' , output_dir
                            ], namespace=args)
    args.n_gpu = 1
    return args

def test_single_aug_addition():
    print('=========Testing Single Aug Addition=========')
    args = get_args()
    # Test function
    new_f_name = get_aug_filename(args)
    add_aug(args, new_f_name)

    # Check file exists & Delete
    print(f'{new_f_name} Exists: {os.path.exists(new_f_name)}')

def generate_all_single_aug_exp_data(squad_path):
    augs_names = [augs_names = ['insert-word-embed', 'sub-word-embed', 'insert-bert-embed','sub-bert-embed', 'delete-random']]
    aug_count = 4
    exp_names = [f'{x}_{aug_count}-count' for x in augs_names]
    for exp_name in exp_names:
        print(f'Generating augs exp: {exp_name}')
        for aug in tqdm(augs_names, desc='Augs'):
            # open folder for expirement
            output_dir = f'{squad_path}/{exp_name}/{aug}'
            os.mkdir(output_dir)
            for seed in tqdm([42,43,44,45,46], desc='Seeds'):
                for num_examples in tqdm([16,32,64,128,256], desc='Examples Num'):
                    train_file_name = f'squad-train-seed-{seed}-num-examples-{num_examples}.jsonl'
                    train_file = f'{squad_path}/{train_file_name}'

                    # Gets args and add augmentations
                    args = get_args(seed=str(seed),train_file=train_file, augs_names=aug, augs_count=str(aug_count), output_dir=output_dir)
                    new_f_name = get_aug_filename(args)
                    add_aug(args, f'{output_dir}/{new_f_name}')

# insert and sub -> check for num_examples


def verify_same_qid_used():
    for seed in [42, 43, 44, 45, 46]:

        train_file_name = f'squad-train-seed-{seed}-num-examples-16.jsonl'
        cur_file = f'squad/{train_file_name}'
        with open(cur_file, "r", encoding="utf-8") as reader:
            cur_data = [json.loads(line) for line in reader]
        cur_ids = set([line['qas'][0]['id'] for line in cur_data[1:]])
        cur_qids = set([line['qas'][0]['qid'] for line in cur_data[1:]])

        for num_examples in [32, 64, 128, 256, 512]:
            train_file_name = f'squad-train-seed-{seed}-num-examples-{num_examples}.jsonl'
            next_file = f'squad/{train_file_name}'
            with open(next_file, "r", encoding="utf-8") as reader:
                next_data = [json.loads(line) for line in reader]
            next_ids = set([line['qas'][0]['id'] for line in next_data[1:]])
            next_qids = set([line['qas'][0]['qid'] for line in next_data[1:]])

            # make sure they are the same
            print(f'Comparing {cur_file}:{next_file}')
            intersect_ids = cur_ids.intersection(next_ids)
            intersect_qids = cur_qids.intersection(next_qids)
            if (len(cur_ids) - len(intersect_ids) > 0):
                print('Mismatched ids', len(cur_ids - intersect_ids), 'Mismatched qids', len(cur_qids - intersect_qids))

            cur_ids = next_ids
            cur_qids = next_qids
            cur_file = next_file

def generate_data_all_exp(squad_path):
    generate_all_single_aug_exp_data(squad_path)

if __name__ == '__main__':
    # verify_same_qid_used()
    generate_all_single_aug_exp_data('squad')