import argparse
from tqdm import tqdm
import re
import os
import json
import pandas as pd
from collections import Counter

pd.set_option('display.max_rows', 800)
pd.set_option('display.max_columns', 800)
pd.set_option('display.width', 1000)

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exps", "-e",
        nargs='+',
        required=True,
        help="Expirement path to results",
    )
    return parser.parse_args()


def get_qa_res_df(results_path):
    df_all = pd.DataFrame()
    for exp in tqdm(os.listdir(results_path), desc='Expirements'):
        exp_path = f'{results_path}/{exp}'
        dataset = exp.split('-')[0]
        aug = '_'.join(exp.split('-')[1:])
        for num_examples in [16, 32, 64, 128, 256, 512, 1024]:
            for seed in [42, 43, 44, 45, 46]:
                entery = {'dataset': dataset, 'aug': aug, 'examples': num_examples, 'seed': seed, 'EM': None, 'f1': None}
                res_folder_path = f'{exp_path}/output-{num_examples}-{seed}'
                if os.path.exists(res_folder_path):
                    if 'eval_results.txt' in os.listdir(res_folder_path):
                        res_file = f'{res_folder_path}/eval_results.txt'
                        with open(res_file, "r") as f:
                            lines = f.readlines()

                        f1 = re.findall("\d+\.\d+", [x for x in lines if x.startswith('best_f1 = ')][0])[0]
                        em = re.findall("\d+\.\d+", [x for x in lines if x.startswith('best_exact = ')][0])[0]
                        entery = {'dataset': dataset, 'aug': aug, 'examples': num_examples, 'seed': seed, 'EM': float(em), 'f1': float(f1)}

                df_all = df_all.append(entery, ignore_index=True)
    return df_all

def average_seeds(df):

    averages_df = pd.DataFrame()
    for dataset in df['dataset'].unique():
        for aug in df['aug'].unique():
            for examples in df['examples'].unique():
                ds_aug_ex_df = df[(df['examples'] == examples) & (df['aug'] == aug) & (df['dataset'] == dataset)]
                df_exp_examples_mean = ds_aug_ex_df.mean(axis=0)
                df_exp_examples_mean['dataset'] = dataset
                df_exp_examples_mean['aug'] = aug
                df_exp_examples_mean['examples'] = examples

                df_exp_examples_var = ds_aug_ex_df.std(axis=0)
                naming_dict = {x: x + '_var' for x in df_exp_examples_var.keys()}
                df_exp_examples_var = df_exp_examples_var.rename(naming_dict)

                df_mean_var = pd.concat([df_exp_examples_mean, df_exp_examples_var], axis=0)

                averages_df = averages_df.append(df_mean_var, ignore_index=True)
    return averages_df

def average_datasets(df):
    averages_df = pd.DataFrame()
    #for dataset in df['dataset'].unique(): #TODO:when missing expirements run change back to this
    df = df[[x in ['naturalquestions', 'bioasq', 'hotpotqa', 'newsqa'] for x in df['dataset']]]
    for aug in df['aug'].unique():
        for examples in df['examples'].unique():
            ds_aug_ex_df = df[(df['examples'] == examples) & (df['aug'] == aug)]
            df_exp_examples_mean = ds_aug_ex_df.mean(axis=0)
            df_exp_examples_mean['aug'] = aug
            df_exp_examples_mean['examples'] = examples
            averages_df = averages_df.append(df_exp_examples_mean, ignore_index=True)
    return averages_df


def calc_diff(v_base, v_new):
    diff_val = round(float(v_new) - float(v_base), 3)
    relative_diff_val = round(diff_val * 100 / float(v_base), 1)
    if diff_val >= 0:
        diff_return_str = str(f'+{diff_val}(+{relative_diff_val}\\%)')
    else:
        diff_return_str = str(f'{diff_val}({relative_diff_val}\\%)')
    return diff_return_str

def delta_from_baseline(df):
    delta_df = pd.DataFrame()
    for examples in df['examples'].unique():
        baseline = df[(df['examples'] == examples) & (df['aug'] == 'baseline')]
        for aug in set(df['aug'].unique()) - set(['baseline']):
            aug_df = df[(df['examples'] == examples) & (df['aug'] == aug)]
            delta_dict = {'examples': examples,
                          'aug': aug,
                          'EM': calc_diff(baseline['EM'], aug_df['EM']),
                          'f1': calc_diff(baseline['f1'], aug_df['f1'])}
            delta_df = delta_df.append(delta_dict, ignore_index=True)
    delta_df = delta_df.round(3)
    return delta_df

def print_overleaf_style(df):
    if 'dataset' in df.columns:
        for dataset in df['dataset'].unique():
            for ex_num in df['examples'].unique():
                for aug in df['aug'].unique():
                    row = df[(df['aug']==aug) & (df['dataset'] == dataset) & (df['examples'] == ex_num)]
                    latex_line = f"\\verb|{row['dataset'].values[0]}| & {row['examples'].values[0]} & \\verb|{row['aug'].values[0]}| & {row['EM'].values[0]} & {row['f1'].values[0]}\\\\"
                    print(latex_line)
                print('\hline')
    else:
        for ex_num in df['examples'].unique():
            for aug in df['aug'].unique():
                row = df[(df['aug']==aug) & (df['examples'] == ex_num)]
                latex_line = f"{row['examples'].values[0]} & \\verb|{row['aug'].values[0]}| & {row['EM'].values[0]} & {row['f1'].values[0]}\\\\"
                print(latex_line)
            print('\hline')

def print_overleaf_style_mean_var_datasets(df):
    # aug_name_dict = {'double_baseline':'baseline-double', 'baseline':'baseline', 'lorem_ipsum':'lorem-ipsum','mosaic':'mosaic-concat','mosaic_crop':'mosaic'}
    # try:
    #     df['aug'] = [aug_name_dict['_'.join(x.split("-")[1:])] for x in df['exp']]
    # except:
    #     import pdb; pdb.set_trace()
    # df['dataset'] = [x.split("-")[0] for x in df['exp']]
    df = df.round(3)
    for d in df['dataset'].unique():
        print('\hline \hline \multicolumn{6}{c}{\\textbf{' + d + '}} \\\\')
        for ex_num in df['examples'].unique():
            print('\hline')

            for aug in sorted(df['aug'].unique()):

                row = df[(df['dataset']==d) & (df['aug']==aug) & (df['examples'] == ex_num)]
                baseline_row = df[(df['dataset']==d) & (df['aug']=='baseline') & (df['examples'] == ex_num)]

                latex_lines = [f"{row['examples'].values[0]} & {row['aug'].values[0]} &"]
                for metric in ['EM','f1']:
                    # LINE1
                    latex_line = f" {row[metric].values[0]:.3f} $\pm$ {row[metric + '_var'].values[0]:.3f} &"
                    # if baseline_row[metric].values[0] >= row[metric].values[0]:
                    #     latex_line = f" {row[metric].values[0]:.3f} $\pm$ {row[metric+'_var'].values[0]:.3f} &"
                    # else:
                    #     latex_line = f" \\textbf{{{row['accuracy'].values[0]:.3f}}} $\pm$ \\textbf{{{row['accuracy_var'].values[0]:.3f}}} &"
                    latex_lines.append(latex_line)

                print(''.join(latex_lines)[:-1] + '\\\\')

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

def get_missing_expirements(df):
    print(df[df['f1'].isna()])



if __name__ == '__main__':

    # PAPER RES
    results_path = '/cs/labs/gabis/ednussi/splinter/finetuning/results'
    df = get_qa_res_df(results_path)
    df = df[df['examples'] <= 256]
    df = df.loc[df['dataset'].isin(['squad', 'hotpotqa'])]
    df = df.loc[df['aug'].isin(['baseline', 'mosaic_2_False', 'mosaiccrop', 'lorem_ipsum'])]
    miss_tot = df['f1'].isnull().sum()
    print(f'Missing total: {miss_tot}')

    avg_seed_df = average_seeds(df)
    ### Table 1
    print_overleaf_style_mean_var_datasets(avg_seed_df)

    import pdb; pdb.set_trace()
    ### Table 2
    avg_seed_ds_df = average_datasets(avg_seed_df)
    delta_df = delta_from_baseline(avg_seed_df)
    print_overleaf_style(delta_df)






    results_path = '/cs/labs/gabis/ednussi/splinter/finetuning/results'
    df = get_qa_res_df(results_path)
    df = df[df['examples'] <= 256]
    df = df.loc[df['dataset'].isin(['squad', 'bioasq', 'hotpotqa', 'newsqa', 'naturalquestions'])]
    df = df.loc[df['aug'].isin(['baseline', 'mosaic_2_False', 'mosaiccrop', 'lorem_ipsum'])]
    miss_tot = df['f1'].isnull().sum()
    print(f'Missing total: {miss_tot}')

    #nothing searchqa triviaqa
    all_datasets =['searchqa', 'newsqa', 'naturalquestions', 'hotpotqa', 'bioasq',
       'triviaqa', 'squad', 'textbookqa']

    all_aug = ['baseline', 'mosaic_2_False', 'coherent_text',
     'lorem_ipsum_double', 'mosaiccrop_2_False', 'concat_coherent_text',
     'mosaic_2_True', 'mosaiccrop', 'lorem_ipsum']

    for d in ['squad', 'bioasq', 'hotpotqa', 'newsqa', 'naturalquestions']:
        # check how muhc is missing
        cond_df = df[df['dataset'] == 'squad']
        cond_df.isnull().sum()
        miss_d = cond_df.isnull().sum()
        print(f'Missing for {d} total: {miss_d}')

        print(f'===== {d} =====')
        avg_seed_df = average_seeds(cond_df)
        print_overleaf_style(avg_seed_df)

    import pdb; pdb.set_trace()

    # check how muhc is missing
    cond_df = df[df['dataset']=='naturalquestions']
    cond_df
    cond_df.isnull().sum()

    avg_seed_df = average_seeds(cond_df)
    print_overleaf_style(avg_seed_df)
    avg_seed_ds_df = average_datasets(avg_seed_df)
    delta_df = delta_from_baseline(avg_seed_df)
    print("Deltas df\n")
    print_overleaf_style(delta_df)
    import pdb; pdb.set_trace()

    #ALL
    avg_seed_df = average_seeds(df)
    avg_seed_ds_df = average_datasets(avg_seed_df)
    delta_df = delta_from_baseline(avg_seed_ds_df)
    print_overleaf_style(delta_df)

    #
    cond_df = df[df['dataset'] == 'hotpotqa']
    cond_df = cond_df[cond_df['examples'] <= 256]
    cond_df['f1'].isnull().sum()