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
                res_file = f'{outputs_path}/{res_folder_path}/eval_results.txt'
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

def plot_f1_em_dict(res_dict):

    fig, ax_f1, ax_em = init_plot()

    line_name = 'splinter_base'
    x = [16,32,64,128] #256
    y_f1 = []
    y_em = []
    yerr_f1_min = []
    yerr_em_min = []
    yerr_f1_max = []
    yerr_em_max = []

    for k in x:


        k_runs = [x for x in res_dict.keys() if x.startswith(str(k))]
        f1s = np.array([float(res_dict[run]['f1'][0]) for run in k_runs])
        ems = np.array([float(res_dict[run]['exact'][0]) for run in k_runs])

        if len(f1s) + len(ems) == 0:
            f1s = np.array([0] * len(x))
            ems = np.array([0] * len(x))

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


def plot_f1_em_dicts(names, dicts):

    fig, ax_f1, ax_em = init_plot()

    for line_name, d in zip(names, dicts):
        x = [16,32,64,128] #256
        y_f1 = []
        y_em = []
        yerr_f1_min = []
        yerr_em_min = []
        yerr_f1_max = []
        yerr_em_max = []

        for k in x:


            k_runs = [x for x in d.keys() if x.startswith(str(k))]
            f1s = np.array([float(d[run]['f1'][0]) for run in k_runs])
            ems = np.array([float(d[run]['exact'][0]) for run in k_runs])

            if len(f1s) + len(ems) == 0:
                f1s = np.array([0] * len(x))
                ems = np.array([0] * len(x))

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


def get_f1_em_dict_num_aug_exp():
    outputs_path = '/cs/labs/gabis/ednussi/splinter/finetuning/outputs/num_augs_exp'

    for aug in ['sub-word', 'insert-bert']:

        res_dict = {}
        for num_augs in [1, 2, 3]:
            for num_examples in tqdm([16, 32, 64, 128, 256], desc='Examples Num'):

                for seed in tqdm([42, 43, 44, 45, 46], desc='Seeds'):

                    res_folder_path = f'output-{aug}-{num_examples}-{seed}-{seed}-{num_augs}'
                    res_file = f'{outputs_path}/{res_folder_path}/eval_results.txt'
                    if os.path.exists(res_file):
                        with open(res_file, "r") as f:
                            lines = f.readlines()

                        f1 = re.findall("\d+\.\d+", [x for x in lines if x.startswith('best_f1 = ')][0])
                        em = re.findall("\d+\.\d+", [x for x in lines if x.startswith('best_exact = ')][0])
                        res_dict[f'{num_examples}-{seed}'] = {'exact':em, 'f1':f1}

        # plot this aug
        print(aug)
        print(res_dict)

def plot_june_ninth_res():
    return

def plot_june_first_res():
    insert_word = {'16-42': {'exact': ['4.207'], 'f1': ['6.516']}, '16-43': {'exact': ['5.730'], 'f1': ['9.978']},
     '16-44': {'exact': ['9.042'], 'f1': ['16.027']}, '16-45': {'exact': ['2.170'], 'f1': ['5.289']},
     '16-46': {'exact': ['5.768'], 'f1': ['7.932']}, '32-42': {'exact': ['6.424'], 'f1': ['10.346']},
     '32-43': {'exact': ['14.067'], 'f1': ['20.653']}, '32-44': {'exact': ['19.844'], 'f1': ['28.444']},
     '32-45': {'exact': ['15.228'], 'f1': ['20.827']}, '32-46': {'exact': ['15.333'], 'f1': ['22.139']},
     '64-42': {'exact': ['23.156'], 'f1': ['29.377']}, '64-43': {'exact': ['26.801'], 'f1': ['35.368']},
     '64-44': {'exact': ['26.754'], 'f1': ['35.817']}, '64-45': {'exact': ['10.279'], 'f1': ['13.740']},
     '64-46': {'exact': ['24.831'], 'f1': ['31.113']}, '128-42': {'exact': ['32.521'], 'f1': ['41.913']},
     '128-43': {'exact': ['36.442'], 'f1': ['43.836']}, '128-44': {'exact': ['44.028'], 'f1': ['52.476']},
     '128-45': {'exact': ['34.253'], 'f1': ['40.423']}, '128-46': {'exact': ['39.907'], 'f1': ['47.744']}}

    sub_word = {'16-42': {'exact': ['3.569'], 'f1': ['5.851']}, '16-43': {'exact': ['4.930'], 'f1': ['8.387']}, '16-44': {'exact': ['5.568'], 'f1': ['9.988']}, '16-45': {'exact': ['1.770'], 'f1': ['4.554']}, '16-46': {'exact': ['4.112'], 'f1': ['6.375']}, '32-42': {'exact': ['8.214'], 'f1': ['12.589']}, '32-43': {'exact': ['10.593'], 'f1': ['16.887']}, '32-44': {'exact': ['20.367'], 'f1': ['29.290']}, '32-45': {'exact': ['10.707'], 'f1': ['15.389']}, '32-46': {'exact': ['14.942'], 'f1': ['21.200']}, '64-42': {'exact': ['17.255'], 'f1': ['21.947']}, '64-43': {'exact': ['25.193'], 'f1': ['33.124']}, '64-44': {'exact': ['22.395'], 'f1': ['30.204']}, '64-45': {'exact': ['17.331'], 'f1': ['21.969']}, '64-46': {'exact': ['26.068'], 'f1': ['33.391']}, '128-42': {'exact': ['26.677'], 'f1': ['32.834']}, '128-43': {'exact': ['34.415'], 'f1': ['41.417']}, '128-44': {'exact': ['39.250'], 'f1': ['47.647']}, '128-45': {'exact': ['17.588'], 'f1': ['20.865']}, '128-46': {'exact': ['40.992'], 'f1': ['49.047']}}
    insert_bert = {'16-42': {'exact': ['5.644'], 'f1': ['8.980']}, '16-43': {'exact': ['3.065'], 'f1': ['4.922']}, '16-44': {'exact': ['9.299'], 'f1': ['16.657']}, '16-45': {'exact': ['1.428'], 'f1': ['3.905']}, '16-46': {'exact': ['4.311'], 'f1': ['5.838']}, '32-42': {'exact': ['7.757'], 'f1': ['11.775']}, '32-43': {'exact': ['9.965'], 'f1': ['15.500']}, '32-44': {'exact': ['13.962'], 'f1': ['18.250']}, '32-45': {'exact': ['12.097'], 'f1': ['17.732']}, '32-46': {'exact': ['13.677'], 'f1': ['19.597']}, '64-42': {'exact': ['19.730'], 'f1': ['25.524']}, '64-43': {'exact': ['27.287'], 'f1': ['35.399']}, '64-44': {'exact': ['26.877'], 'f1': ['35.298']}, '64-45': {'exact': ['16.180'], 'f1': ['20.604']}, '64-46': {'exact': ['27.401'], 'f1': ['35.221']}, '128-43': {'exact': ['38.184'], 'f1': ['46.511']}, '128-44': {'exact': ['44.884'], 'f1': ['53.927']}, '128-45': {'exact': ['34.053'], 'f1': ['40.989']}, '128-46': {'exact': ['43.590'], 'f1': ['51.693']}}
    sub_bert = {'16-42': {'exact': ['3.483'], 'f1': ['5.999']}, '16-43': {'exact': ['3.883'], 'f1': ['7.038']}, '16-45': {'exact': ['1.351'], 'f1': ['3.638']}, '16-46': {'exact': ['3.845'], 'f1': ['5.954']}, '32-42': {'exact': ['5.206'], 'f1': ['8.827']}, '32-43': {'exact': ['8.699'], 'f1': ['14.026']}, '32-44': {'exact': ['16.608'], 'f1': ['24.796']}, '32-45': {'exact': ['6.748'], 'f1': ['10.752']}, '32-46': {'exact': ['11.116'], 'f1': ['16.437']}, '64-42': {'exact': ['15.247'], 'f1': ['19.749']}, '64-43': {'exact': ['19.359'], 'f1': ['26.196']}, '64-44': {'exact': ['21.852'], 'f1': ['29.529']}, '64-45': {'exact': ['19.330'], 'f1': ['24.742']}, '64-46': {'exact': ['22.014'], 'f1': ['30.077']}, '128-42': {'exact': ['29.704'], 'f1': ['38.446']}, '128-43': {'exact': ['29.656'], 'f1': ['36.370']}, '128-44': {'exact': ['39.916'], 'f1': ['48.002']}, '128-45': {'exact': ['10.488'], 'f1': ['12.426']}, '128-46': {'exact': ['37.803'], 'f1': ['45.590']}}

    names = ['insert_word', 'sub_word', 'insert_bert', 'sub_bert']
    dicts = [insert_word, sub_word, insert_bert, sub_bert]
    plot_f1_em_dicts(names, dicts)

if __name__ == '__main__':
    # outputs_path = 'outputs'
    # get_f1_em_dict(outputs_path)
    get_f1_em_dict_num_aug_exp()
    # plot_june_first_res()
