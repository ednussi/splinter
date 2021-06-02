# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# import torch
from tqdm import tqdm
# from transformers import AdamW, create_optimizer, get_linear_schedule_with_warmup
# from torch.utils.data import DataLoader
# import math
# import pandas as pd
# import json
# import sys
# import string
# import tempfile
# import subprocess
import re
import matplotlib.pyplot as plt
# from data.squad.eval1 import evaluate
# import nlpaug.augmenter.word as naw
import os
# import requests
# import gzip
# import shutil
import numpy as np


# # ============ Save ============
# # save answers df
# answers_df_name = f'{base_path}/results/{exp_name}/data/answers_{exp_params}.csv'
# answers_df.to_csv(answers_df_name)
# EM, F1 = [x.split(':')[1] for x in res_summary.split(',')]
# with open(results_path, "a") as myfile:
#     save_string = f'{csv_entery_num},{EM},{F1},{seed},{shuffle_seed},{n},{base_model}\n'
#     myfile.write(save_string)
# csv_entery_num += 1
#
# # ============ Plot Results ============
# if args.plot_res:
#     plot_random_sample_res([results_path], exp_name, base_path)


def get_f1_em_dict(outputs_path):

    for aug in ['insert-word', 'sub-word', 'insert-bert','sub-bert', 'delete-random']:

        res_dict = {}
        for num_examples in tqdm([16, 32, 64, 128, 256], desc='Examples Num'):

            for seed in tqdm([42, 43, 44, 45, 46], desc='Seeds'):
                res_folder_path = f'output-{aug}-{num_examples}-{seed}'
                res_file = f'{res_folder_path}/eval_results.txt'
                if os.path.exists(res_file):
                    with open(res_file, "r") as f:
                        lines = f.readlines()

                    f1 = re.findall("\d+\.\d+", [x for x in lines if x.startswith('best_f1 = ')][0])
                    em = re.findall("\d+\.\d+", [x for x in lines if x.startswith('best_exact = ')][0])
                    res_dict[f'{num_examples}-{seed}'] = {'exact':em, 'f1':f1}

        # plot this aug
        print(aug)
        print(res_dict)
        #plot_f1_em_dict(res_dict)

def init_plot():
    plt.figure(figsize=(20,30))
    fig, (ax_f1, ax_em) = plt.subplots(2)
    ax_em.set_xscale("log")
    ax_f1.set_xscale("log")
    fig.suptitle('Experiment Results')
    return fig, ax_f1, ax_em

def plot_f1_em_dict():

    fig, ax_f1, ax_em = init_plot()

    res_dict = {'64-43': {'exact': 23.289, 'f1': 31.024}, '16-46': {'exact': 4.064, 'f1': 5.716},
                '16-43': {'exact': 5.511, 'f1': 9.563}, '64-45': {'exact': 22.004, 'f1': 28.248},
                '32-42': {'exact': 7.052, 'f1': 11.504}, '256-46': {'exact': 47.464, 'f1': 57.048},
                '128-43': {'exact': 30.494, 'f1': 39.791}, '64-46': {'exact': 25.136, 'f1': 33.040},
                '128-44': {'exact': 35.719, 'f1': 43.534}, '32-44': {'exact': 19.378, 'f1': 27.676},
                '32-43': {'exact': 11.402, 'f1': 17.522}, '64-42': {'exact': 21.662, 'f1': 26.912},
                '16-44': {'exact': 9.089, 'f1': 16.428}, '16-42': {'exact': 4.683, 'f1': 6.779},
                '256-43': {'exact': 52.213, 'f1': 61.855}, '256-45': {'exact': 40.973, 'f1': 50.231},
                '128-45': {'exact': 29.219, 'f1': 35.873}, '256-42': {'exact': 41.515, 'f1': 50.887},
                '16-45': {'exact': 1.713, 'f1': 4.461}, '128-46': {'exact': 39.688, 'f1': 49.168},
                '64-44': {'exact': 25.707, 'f1': 33.754}, '256-44': {'exact': 48.568, 'f1': 58.361},
                '128-42': {'exact': 27.163, 'f1': 34.966}}

    line_name = 'splinter_base'
    x = [16,32,64,128,256]
    y_f1 = []
    y_em = []
    yerr_f1_min = []
    yerr_em_min = []
    yerr_f1_max = []
    yerr_em_max = []

    for k in x:
        k_runs = [x for x in res_dict.keys() if x.startswith(str(k))]
        f1s = np.array([res_dict[run]['f1'] for run in k_runs])
        ems = np.array([res_dict[run]['exact'] for run in k_runs])

        y_f1.append(f1s.mean())
        y_em.append(ems.mean())

        yerr_f1_min.append(f1s.min())
        yerr_em_min.append(ems.min())
        yerr_f1_max.append(f1s.max())
        yerr_em_max.append(ems.max())

    ax_f1.plot(x, y_f1, label=line_name)
    ax_f1.fill_between(x, yerr_f1_min, yerr_f1_max, alpha=0.5)
    ax_em.plot(x, y_em, label=line_name)
    ax_em.fill_between(x, yerr_em_min, yerr_em_max, alpha=0.5)

    xrange = [2 ** x for x in range(4, 9)]
    xrange_text = [str(x) for x in xrange]

    ax_f1.set_title('F1 vs. #QA pairs')
    ax_f1.legend(prop={'size':6}, loc='upper left')
    ax_f1.set_xticks(xrange)
    ax_f1.set_xticklabels(xrange_text)
    ax_em.set_title('EM vs. #QA pairs')
    ax_em.legend(prop={'size':6}, loc='upper left')
    ax_em.set_xticks(xrange)
    ax_em.set_xticklabels(xrange_text)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    outputs_path = 'outputs'
    get_f1_em_dict(outputs_path)
