import argparse
from tqdm import tqdm
import re
import os
import json
from pandas.io.json import json_normalize
import pandas as pd
import time
def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exps", "-e",
        nargs='+',
        required=True,
        help="Expirement path to results",
    )
    return parser.parse_args()

def get_results_df(ner_results_path):
    columns = ['exp', 'examples', 'seed', 'f1', 'accuracy', 'precision', 'recall', 'loss']
    df_all = pd.DataFrame(columns=columns)

    for exp in os.listdir(ner_results_path):
        exp_path = f'{ner_results_path}/{exp}'
        for num_examples in tqdm([16, 32, 64, 128, 256, 512, 1024], desc='Examples'):
            for seed in tqdm([42, 43, 44, 45, 46], desc='Seeds'):
                res_folder_path = f'{exp_path}/output-{num_examples}-{seed}'
                if 'eval_results.json' in os.listdir(res_folder_path):
                    res_file = f'{res_folder_path}/eval_results.json'
                    with open(res_file, "r") as f:
                        data = json.load(f)
                    res_dict = {'exp':exp, 'examples': num_examples, 'seed': seed, 'f1': data['eval_f1'], 'accuracy': data['eval_accuracy'],
                                                          'precision': data['eval_precision'], 'recall': data['eval_recall'],
                                                          'loss': data['eval_loss']}
                    df_all = df_all.append(res_dict, ignore_index=True)

    return df_all
if __name__ == '__main__':
    # args = init_parser()
    ner_results_path = 'results_ner'
    ner_results_path = 'ner_res'
    get_results_df(ner_results_path)