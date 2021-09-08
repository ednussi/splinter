import json
import pandas as pd
from tqdm import tqdm
import os
import time
# from spacy.lang.en import English

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
                import pdb;pdb.set_trace()
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

def qas_clique_unite(df):
    united_df = pd.DataFrame()
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

def qas_npairs_unite(df, pairs):
    # Divide and runs in clique - aggrigate in end
    npairs_df = pd.DataFrame()
    for df_chunk in split_dataframe(df, chunk_size=pairs):
        united_df = qas_clique_unite(df_chunk)
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

def print_df_example(df, index=0):
    row = df.iloc[index]
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

def mosaic_npairs_single_qac_aug(input_data, pairs=2, final_single_qac_triplets=True):
    df = input_data_to_df(input_data)
    split_df = split_qas_to_single_qac_triplets(df)
    uni_df = qas_npairs_unite(split_df, pairs)
    if final_single_qac_triplets:
        uni_single_qac_df = split_qas_to_single_qac_triplets(uni_df)
        return uni_single_qac_df
    return uni_df

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