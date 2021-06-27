import json
import pandas as pd
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
import os
# from spacy.lang.en import English

def train_file_to_df(train_file_path):
    # open file
    with open(train_file_path, "r", encoding="utf-8") as reader:
        input_data = [json.loads(line) for line in reader]

    df = pd.DataFrame()  # columns=['id','qid','answers','question','question_tokens','detected_answers']

    for line in tqdm(input_data[1:], desc='Enhancing with augs'):
        df = df.append(line, ignore_index=True)

    df['qas']

    return df


def split_qas_to_single_qac_triplets(df):
    split_df = pd.DataFrame()
    for i, row in df.iterrows():
        if len(row['qas']) > 1:
            for qas_triplet in row['qas']:
                split_df = split_df.append([row])
        else:
            split_df = split_df.append(row)
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
        row1_context_length = len(de1['context'])
        row2_updated_qas['detected_answers'][det_ans_i]['char_spans'] = [
            [x[0] + row1_context_length + 1, x[1] + row1_context_length + 1] for x in
            row2_updated_qas['detected_answers'][det_ans_i]['char_spans']]
        row2_updated_qas['detected_answers'][det_ans_i]['token_spans'] = [
            [x[0] + len(de1['context_tokens']), x[1] + len(de1['context_tokens'])] for x in
            row2_updated_qas['detected_answers'][det_ans_i]['token_spans']]
    combined_qas = [de1['qas'][0], row2_updated_qas]
    return combined_qas, context_tokens, combined_context, combined_id

def qas_pairs_unite(df):
    united_df = pd.DataFrame()
    for i in range(0, len(df), 2):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]
        # insert in both regular order and oppisite in concatination
        combined_qas, context_tokens, combined_context, combined_id = get_combined_de(row1, row2)
        united_df = united_df.append({'id':combined_id,
                                      'context':combined_context,
                                      'context_tokens':context_tokens,
                                      'qas':combined_qas},ignore_index=True)

        combined_qas, context_tokens, combined_context, combined_id = get_combined_de(row2, row1)
        united_df = united_df.append({'id':combined_id,
                                      'context':combined_context,
                                      'context_tokens':context_tokens,
                                      'qas':combined_qas},ignore_index=True)

    return united_df

# def window_example


def write_df(df, name):
    new_jsonl_lines = []
    header = {"header": {"dataset": "SQuAD", "split": "train"}}

    for i,row in df.iterrows():
        new_jsonl_lines.append(row.to_dict())

    # Write as a new jsonl file
    logger.info(f'Writing new augmented data file to {name}')
    print(f'Writing new augmented data file to {name}')
    with open(name, "w", encoding="utf-8") as writer:
        writer.write(f'{json.dumps(header)}\n')
        for line in new_jsonl_lines:
            writer.write(f'{json.dumps(line)}\n')
    return


def create_moasic_unite_exp_data(squad_path):
    exp_name = 'moasic_unite'
    # open folder for expirement
    output_dir = f'{squad_path}/{exp_name}'
    os.mkdir(output_dir)
    for seed in tqdm([42, 43, 44, 45, 46], desc='Seeds'):
        for num_examples in tqdm([16, 32, 64, 128, 256], desc='Examples Num'):
            train_file_name = f'squad-train-seed-{seed}-num-examples-{num_examples}.jsonl'
            df = train_file_to_df(f'{squad_path}/{train_file_name}')
            df = split_qas_to_single_qac_triplets(df)
            uni_df = qas_pairs_unite(df)
            write_df(uni_df, f'{output_dir}/squad-train-seed-{seed}-num-examples-{num_examples}.jsonl')

if __name__ == '__main__':
    squad_path = 'squad'
    create_moasic_unite_exp_data(squad_path)