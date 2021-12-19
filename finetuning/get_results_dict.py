import argparse
from tqdm import tqdm
import re
import os
import json
import pandas as pd

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exps", "-e",
        nargs='+',
        required=True,
        help="Expirement path to results",
    )
    return parser.parse_args()


def get_qa_res_df():
    results_path = '/cs/labs/gabis/ednussi/splinter/finetuning/results'
    df_all = pd.DataFrame()
    for exp in os.listdir(results_path):
        exp_path = f'{results_path}/{exp}'
        for num_examples in tqdm([16, 32, 64, 128, 256, 512, 1024], desc='Examples'):
            for seed in tqdm([42, 43, 44, 45, 46], desc='Seeds'):
                res_folder_path = f'{exp_path}/output-{num_examples}-{seed}'
                if os.path.exists(res_folder_path):
                    if 'eval_results.txt' in os.listdir(res_folder_path):
                        res_file = f'{res_folder_path}/eval_results.txt'
                        with open(res_file, "r") as f:
                            lines = f.readlines()

                        f1 = re.findall("\d+\.\d+", [x for x in lines if x.startswith('best_f1 = ')][0])
                        em = re.findall("\d+\.\d+", [x for x in lines if x.startswith('best_exact = ')][0])
                        entery = {'dataset': None, 'aug': None, 'examples': num_examples, 'seed': seed, 'EM': em, 'f1': f1}

                    elif 'eval_results.json' in os.listdir(res_folder_path):
                        import pdb; pdb.set_trace()
                        res_file = f'{res_folder_path}/eval_results.json'
                        with open(res_file, "r") as f:
                            data = json.load(f)

                        entery = {'dataset': None, 'aug': None, 'examples': num_examples, 'seed': seed, 'EM': em, 'f1': f1}
                    df_all = df_all.append(df_all, ignore_index=True)
    return df_all

def get_f1_em_dict(exp_paths):
    base_path = os.getcwd()
    for exp in exp_paths:

        res_dict = {}
        for num_examples in tqdm([16, 32, 64, 128, 256], desc='Base Examples'):

            for seed in tqdm([42, 43, 44, 45, 46], desc='Seeds'):

                res_folder_path = f'{exp}/output-{num_examples}-{seed}'
                if 'eval_results.txt' in os.listdir(f'{base_path}/{res_folder_path}'):

                    res_file = f'{base_path}/{res_folder_path}/eval_results.txt'
                    with open(res_file, "r") as f:
                        lines = f.readlines()

                    f1 = re.findall("\d+\.\d+", [x for x in lines if x.startswith('best_f1 = ')][0])
                    em = re.findall("\d+\.\d+", [x for x in lines if x.startswith('best_exact = ')][0])
                    res_dict[f'{num_examples}-{seed}'] = {'exact':em, 'f1':f1}

                elif 'eval_results.json' in os.listdir(f'{base_path}/{res_folder_path}'):
                    res_file = f'{base_path}/{res_folder_path}/eval_results.json'
                    with open(res_file, "r") as f:
                        data = json.load(f)

                        res_dict[f'{num_examples}-{seed}'] = {'exact': em, 'f1': f1}

        # plot this aug
        print(f'============ {exp} ============')
        print(res_dict)

if __name__ == '__main__':
    df = get_qa_res_df()
    import pdb; pdb.set_trace()

    # args = init_parser()
    # get_f1_em_dict(args.exps)