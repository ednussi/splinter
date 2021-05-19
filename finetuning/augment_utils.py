import os
import json
import gzip
import shutil
import nlpaug.augmenter.word as naw
import requests
from tqdm import tqdm
import re
import logging
logger = logging.getLogger(__name__)
from utils import init_parser

from transformers import (
    AutoTokenizer,
    squad_convert_examples_to_features,
)

from modeling import set_seed

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


def get_words_start_pos_list(text, tokenizer):
    encoded_input = tokenizer(text)
    words_tokenized = [tokenizer.decode(token) for token in encoded_input['input_ids']]
    words_tokenized = [w.strip() for w in words_tokenized if w not in ['<s>','</s>']]
    text_tokens = [[word.group(), word.start()] for word in re.finditer(r'\S+', ' '.join(words_tokenized))]
    text_tokens[-1][1] = text_tokens[-1][1] - 1 # question mark indicatng last token?
    return text_tokens


def add_aug(args, new_f_name):
    logger.info(f'Loading Examples from file {args.train_file}')
    # open file
    with open(args.train_file, "r", encoding="utf-8") as reader:
        input_data = [json.loads(line) for line in reader]

    augs = get_augs_from_names(args.augs_names)
    logger.info(f'Loading Tokenizer {args.tokenizer_name}')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, add_prefix_space=True) #add_prefix_space=True for roberta-base

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
            q_tokens = get_words_start_pos_list(q, tokenizer)
            assert(q_tokens == qas['question_tokens'], f'{q_tokens},\n {qas["question_tokens"]} ')

            for aug_num,aug in zip(args.augs_count, augs):
                # create augs per count
                set_seed(args) #initialize seed
                news_q_augs_set = get_n_new_aug(aug, q, int(aug_num))

                for new_aug_q_i, new_aug_q in enumerate(news_q_augs_set):
                    # question/answer/context tokens in the jsonl file are just the position each word starts in
                    new_question_tokens = get_words_start_pos_list(new_aug_q, tokenizer)

                    # append new example
                    augmented_qas = {'answers':answers,
                                           'question':new_aug_q,
                                           'id': f'{id}_{new_aug_q_i}',
                                           'qid': f'{qid}_{new_aug_q_i}',
                                           'question_tokens': new_question_tokens,
                                           'detected_answers': detected_answers}

                    line_copy['qas'] = augmented_qas
                    # Now add all the new questions to this line
                    new_jsonl_lines.append(line_copy)

    # Write as a new jsonl file
    logger.info(f'Writing new augmented data file to {new_f_name}')
    with open(new_f_name, "w", encoding="utf-8") as writer:
        writer.write(f'{json.dumps(header_line)}\n')
        for line in new_jsonl_lines:
            writer.write(f'{json.dumps(line)}\n')
    return

def test_single_aug_addition():
    print('=========Testing Single Aug Addition=========')
    # Create a Namespace immulating argparser
    class C:
        pass
    args = C()
    parser = init_parser()
    parser.parse_args(args=['--seed', '42',
                            '--train_file', 'squad/squad-train-seed-42-num-examples-16.jsonl',
                            '--augs_names', 'sub-bert-embed',
                            '--tokenizer_name', 'roberta-base',
                            '--augs_count', '1',
                            '--model_type', 'roberta-base', # 3 required params
                            '--model_name_or_path', '',
                            '--output_dir' , ''
                            ], namespace=args)
    args.n_gpu = 1
    # Test function
    base_filename = args.train_file.split('augs')[-1].split('.')[0]
    aug_names_count_str = '_'.join([f'{x}-{y}' for x,y in zip(args.augs_names, args.augs_count)])
    new_f_name = f'{base_filename}-augs_{aug_names_count_str}.jsonl'
    add_aug(args, new_f_name)

    # Check file exists & Delete
    print(f'{new_f_name} Exists: {os.path.exists(new_f_name)}')


if __name__ == '__main__':
    test_single_aug_addition()
    #squad_convert_examples_to_features()