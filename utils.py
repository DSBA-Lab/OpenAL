import json
import os
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from collections import defaultdict
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

from query_strategies import get_target_from_dataset

def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, [v]))
    return dict(items)
            
            
def save_full_results(path: str, savedir: str) -> pd.DataFrame:
    """
    Args:
    - path (str): results directory
    - savedir (str): results to save directory
    
    Returns:
    - df (pd.DataFrame): full images supervised learning results
    """
    os.makedirs(savedir, exist_ok=True)
    
    cfg_path = glob(os.path.join(path, '*/Full*/configs.yaml'))

    df = pd.DataFrame()
    for p in cfg_path:
        # results directory
        r_dir = os.path.dirname(p)
        best_path = os.path.join(r_dir, 'results_seed0_best.json')
        test_path = os.path.join(r_dir, 'results-seed0.json')
        
        # unfinished
        if not os.path.isfile(test_path):
            continue
        
        # configs
        cfg = OmegaConf.load(p)
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = flatten_dict(cfg)
        
        # valid best results
        best_r = json.load(open(best_path, 'r'))
        cfg['best_step'] = [best_r.pop('best_step')]
        for k, v in best_r.items():
            cfg[f'best_{k}'] = [v]
        
        # test results
        test_r = json.load(open(test_path,'r'))
        for k, v in test_r.items():
            cfg[f'test_{k}'] = [v]
        
        # add results directory
        cfg['dir'] = r_dir
        
        df = pd.concat([df, pd.DataFrame(cfg)], axis=0)
        
    df.to_csv(os.path.join(savedir, 'full_resutls.csv'), index=False)
    
    
    
def per_class_results(table: dict):
    """
    Confusion matrix and bar-plot of metrics per class
    
    Args:
    - tables (dict): keys in table is confusion matrix and metrics. ex) {'cm': ..., 'acc': ..., } 
    
    Outputs:
    - Confusion matrix
    - bar-plot of metrics per class
    """
    tb = table.copy()
    cm = tb.pop('cm')
    
    # confusion matrix
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='5d', vmin=0, vmax=100, cbar=False)
    ax.set_ylabel('True')
    ax.set_xlabel('Predicsion')
    plt.show()
        
    df_test = pd.DataFrame(tb)
    df_test = df_test.stack().reset_index()
    df_test.rename(columns={'level_0':'class', 'level_1':'Metric', 0:'Score'}, inplace=True)
    df_test['Metric'] = df_test['Metric'].apply(lambda x: x.upper())
    
    # batplot
    fig, ax = plt.subplots(1,1,figsize=(10,3))
    ax = sns.barplot(
        x    = 'Metric',
        y    = 'Score',
        hue  = 'class',
        data = df_test,
        ax   = ax
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', size=10)
    plt.show()
    
def cal_metric_binary(cm: list or np.ndarray, show: bool = True):
    """
    Confusion matrix and metrics calculated as binary class.
    Assume that normal class is 0, the rest are abnormal.
    
    Args:
    - cm (list or np.ndarray): confusion matrix for multi-class
    - show (bool): show confusion matrix figure or not
    
    Return:
    - metrics_bin (np.ndarray): metrics calculated as binary class
    """
    if isinstance(cm, list):
        cm = np.array(cm)
        
    cm_bin = np.zeros((2,2))
    cm_bin[0,0] = cm[0,0]
    cm_bin[0,1] = cm[0,1:].sum()
    cm_bin[1,0] = cm[1:,0].sum()
    cm_bin[1,1] = cm[1:,1:].sum()
    cm_bin = cm_bin.astype(int)
    
    if show:
        # figure confusion matrix
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        sns.heatmap(
            cm_bin, 
            annot       = True, 
            fmt         = '5d', 
            xticklabels = ['OK','NG'], 
            yticklabels = ['OK','NG'], 
            cbar        = False
        )
        ax.set_ylabel('True')
        ax.set_xlabel('Predicsion')
        plt.show()
    
    # calculate matrics
    acc_bin = cm_bin.diagonal().sum() / cm_bin.sum()
    recall_bin = cm_bin[1,1] / cm_bin[1,:].sum()
    precision_bin = cm_bin[1,1] / cm_bin[:,1].sum()
    f1_bin = 2*((recall_bin*precision_bin)/(recall_bin+precision_bin))
    
    metrics_bin = {
        'cm': cm_bin,
        'acc': acc_bin,
        'recall': recall_bin,
        'precision': precision_bin,
        'f1': f1_bin
    }
   
    return metrics_bin



def extract_al_results(
    savedir: str, exp_names: list, seed_list: list, model: str, 
    n_start: int, n_query: int, n_end: int, test: bool = True, binary: bool = False, change_names: dict = None) -> int and dict:
    """
    Extract active learning results
    
    Args:
    - savedir (str): saved result directory
    - exp_names (list): experiment names for active learning
    - seed_list (list): seed list
    - model (str): model name
    - n_start (int): the number of initial samples for active learning
    - n_query (int): the number of query for active learning
    - n_end (int): the number of end samples for active learning
    - test (bool): use test results or not. if not, use best validation results
    - binary (bool): use binary results of not
    - change_names (dict): dictionary for mapping experiment names to new names
    
    Returns:
    - total_round (int): total round for active learning
    - r (dict): active learning results data frame by strategies
    
    =============
    
    Example:
    
    exp_names = [
        'RandomSampling',
        'EntropySampling',
        'MarginSampling',
        'LeastConfidence',
    ]
    seed_list = [0]
    savedir = './results/SamsungAL'
    model = 'swin_base_patch4_window7_224.ms_in22k'
    n_start = 5000
    n_query = 20
    n_end = 6000

    # get results
    total_round, r = extract_al_results(
        savedir   = savedir,
        exp_names = exp_names,
        seed_list = seed_list,
        model     = model,
        n_start   = n_start,
        n_query   = n_query,
        n_end     = n_end,
        test      = test,
        binary    = binary
    )
    """
    
    # total round
    total_round = int((n_end-n_start)/n_query)
    
    # get results
    r = {}
    for i, exp in enumerate(exp_names):

        df_all = pd.DataFrame()

        # concat result dataframe for seed
        for seed in seed_list:
            
            # define filename
            al_filename = f'round{total_round}-seed{seed}_test'
            if not test:
                al_filename += '_valid'
            
            if binary:
                al_filename += '-per_class.json'
            else:
                al_filename += '.csv'            
            
            # get result
            p = os.path.join(
                savedir, model, '*', exp,
                f'total_{n_end}-init_{n_start}-query_{n_query}', f'seed{seed}', al_filename
            )
            try:
                f = glob(p)[0]
            except:
                print(p)
            
            if not binary:
                df_seed = pd.read_csv(f)
                # remove loss
                df_seed = df_seed.drop('loss', axis=1)

            else:
                r_seed_per_class = json.load(open(f, 'r'))
                r_seed_bin = defaultdict(list)
                
                # append binary results per round
                for round_i, m_i in r_seed_per_class.items():
                    # get binary results
                    r_seed_bin_i = cal_metric_binary(m_i['cm'], show=False)
                    # delete confusion matrix of binary class, due to unused key
                    del r_seed_bin_i['cm']
                    # append round i results
                    r_seed_bin['round'].append(int(round_i.strip('round')))
                    for k, v in r_seed_bin_i.items():
                        r_seed_bin[k].append(v)
                        
                # get dataframe
                df_seed = pd.DataFrame(r_seed_bin)
                    
            # add seed column
            df_seed['seed'] = seed
            
            # concate results
            df_all = pd.concat([df_all, df_seed], axis=0)

        # add strategy column
        if change_names is not None:
            if exp in change_names.keys():
                exp = change_names[exp]
        df_all['strategy'] = exp
        r[exp] = df_all
        
    return total_round, r


def extract_full_results(
    savedir: str, fullname_list: list, seed_list: list, model: str,
    test: bool = True, binary: bool = False) -> pd.DataFrame:
    """
    Extract full images supervised learning results
    
    Args:
    - savedir (str): saved result directory
    - fullname_list (list): full images supervised learning experiment foldernames
    - seed_list (list): seed list
    - model (str): model name
    - test (bool): use test results or not. if not, use best validation results
    - binary (bool): use binary results of not
    
    Return:
    - r_full (dict): full images supervised learning results
    
    =============
    
    Example:
    
    fullname_list = [
        'Full-hflip_vflip',
        'Full-hflip_vflip_d-seed43',
        'Full-hflip_vflip_d-seed44',
        'Full-hflip_vflip_d-seed45',
        'Full-hflip_vflip_d-seed46',
    ]
    seed_list = [0]
    savedir = './results/SamsungAL'
    model = 'swin_base_patch4_window7_224.ms_in22k'

    # get full supervised learning results
    r_full = extract_full_results(
        savedir       = savedir,
        fullname_list = fullname_list,
        seed_list     = seed_list,
        model         = model,
        test          = True,
        binary        = False
    )
    """
    
    # get full supervised learning results
    r = {}
    
    # dataset seed
    for f in fullname_list:    
        r_full = defaultdict(list)
        
        # model seed
        for seed in seed_list:
            # define filename
            full_filename = f'results-seed{seed}'
            if not test:
                full_filename += '_valid'
            else:
                full_filename += '_test'
                
            if binary:
                full_filename += '-per_class.json'
            else:
                full_filename += '.json'   
            
            # get results
            full_s = json.load(open(os.path.join(savedir, model, 'Full', f, f'seed{seed}', full_filename), 'r'))
            
            if not test:
                del full_s['best_step']
            
            if binary:
                # get binary results
                full_s = cal_metric_binary(full_s['cm'], show=False)
                # delete confusion matrix of binary class, due to unused key
                del full_s['cm']
                
            for k, v in full_s.items():
                r_full[k].append(v)
        
        r_full = pd.DataFrame(r_full)
        r_full['seed'] = seed_list
        r_full['round'] = 0
        r_full['strategy'] = f
        
        r[f] = r_full
    
    return r


def comparison_strategy(
    savedir: str, exp_names: list, fullname_list: list, seed_list: list, model: str, 
    n_start: int, n_query: int, n_end: int, 
    savepath: str = None, figsize: tuple = (7,10), test: bool = True, binary: bool = False, change_names: dict = None,
    metrics: list = ['acc'], show: bool = True) -> dict:
    """
    Comparison strategies results using figure
    
    Args:
    - savedir (str): saved result directory
    - exp_names (list): experiment names for active learning
    - fullname_list (list): full images supervised learning experiment foldernames
    - seed_list (list): seed list
    - model (str): model name
    - n_start (int): the number of initial samples for active learning
    - n_query (int): the number of query for active learning
    - n_end (int): the number of end samples for active learning
    - test (bool): use test results or not. if not, use best validation results
    - binary (bool): use binary results of not
    - change_names (dict): dictionary for mapping experiment names to new names
    
    Return:
    - r (dict): active learning results data frame by strategies
    
    =============
    
    Example:
    
    exp_names = [
        'RandomSampling',
        'EntropySampling',
        'MarginSampling',
        'LeastConfidence',
    ]
    seed_list = [0]
    savedir = './results/SamsungAL'
    fullname_list = [
        'Full-hflip_vflip',
        'Full-hflip_vflip_d-seed43',
        'Full-hflip_vflip_d-seed44',
        'Full-hflip_vflip_d-seed45',
        'Full-hflip_vflip_d-seed46',
    ]
    model = 'swin_base_patch4_window7_224.ms_in22k'
    n_start = 5000
    n_query = 20
    n_end = 6000

    results = comparison_strategy(
        savedir       = savedir,
        exp_names     = exp_names,
        fullname_list = fullname_list,
        seed_list     = seed_list,
        model         = model,
        n_start       = n_start,
        n_query       = n_query,
        n_end         = n_end,
        test          = True,
        binary        = False,
        figsize       = (17,8)
    )
    """

    # get results
    total_round, r = extract_al_results(
        savedir      = savedir,
        exp_names    = exp_names,
        seed_list    = seed_list,
        model        = model,
        n_start      = n_start,
        n_query      = n_query,
        n_end        = n_end,
        test         = test,
        binary       = binary,
        change_names = change_names
    )
    
            
    # get full supervised learning results
    if fullname_list != None:
        r_full = extract_full_results(
            savedir       = savedir,
            fullname_list = fullname_list,
            seed_list     = seed_list,
            model         = model,
            test          = test,
            binary        = binary
        )
    else:
        r_full = None
        
    # plot
    if show:
        row = 2 if len(metrics) > 1 else 1
        col = len(metrics)//row
        fig, ax = plt.subplots(row, col, figsize=figsize)

        for i in range(row*col):
            if isinstance(ax, np.ndarray):
                if len(ax.shape) == 2:
                    ax_i = ax[i//col, i%col]
                elif (len(ax.shape) == 1) and (len(ax) > 1):
                    ax_i = ax[i]
            else:
                ax_i = ax
                
            strategy_figure(
                data        = r,
                data_full   = r_full,
                metric      = metrics[i],
                total_round = total_round,
                n_query     = n_query,
                n_start     = n_start,
                ax          = ax_i
            )
        
            if i == 0:
                lines, labels = ax_i.get_legend_handles_labels()
                for l in lines:
                    l.set_linewidth(10)
            
            ax_i.get_legend().remove()

        fig.legend(
            lines, labels, ncol=6, loc='lower center', bbox_to_anchor=(0.5, 0.98), 
            frameon=False, fontsize=15, 
        )
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=300)
        plt.show()
            
    
    return r, r_full
    

def strategy_figure(data, data_full, metric, total_round, n_query, n_start, ax):
    data = pd.concat(list(data.values()))

    sns.lineplot(
        x    = 'round',
        y    = metric,
        hue  = 'strategy',
        marker = 'o',
        data = data,
        ax   = ax
    )
    
    # full supervised learning
    if isinstance(data_full, pd.DataFrame):
        ax.axhline(y=data_full[metric].mean(), color='black')
        ax.axhline(y=data_full[metric].mean() + data_full[metric].std(), linestyle='--', color='black')
        ax.axhline(y=data_full[metric].mean() - data_full[metric].std(), linestyle='--', color='black')   
        
    # figure info
    ax.set_ylabel(metric.upper())
    ax.set_xlabel('The Number of Labeled Images')
    ax.set_xticks(
        (np.arange(0,total_round+1,1)).astype(int), 
        (np.arange(0,total_round+1,1)*n_query + n_start).astype(int), 
        rotation = 45, 
        size     = 10
    )

def aubc(scores: list or np.ndarray) -> float:
    """
    Area Under Bucket Curve (AUBC)
    AUBC calculated by Trapezoidal rule
    
    Arg:
    - scores (list or np.ndarray): score per round for AUBC
    
    Return:
    - AURC score
    """
    if type(scores) != np.ndarray:
        scores = np.array(scores)
        
    assert len(scores.shape) == 1, "scores should be 1-d array"
    
    h = 1 / len(scores)
    
    return (scores[0] + (scores[1:-1]*2).sum() + scores[-1])/2 * h
    
def comparison_aubc(results: dict) -> pd.DataFrame:
    """
    Comparison AUBC by strategies
    
    Arg:
    - results (dict): active learning results data frame by strategies
    
    Return:
    - table (pd.DataFrame): AUBC and last metric score by strategies
    
    =============
    
    Example:
    
    # results is output from extract_al_results function
    
    table = comparison_aubc(results)
    table.round(4)
    """
    # define metric for AUBC
    metrics = list(results.values())[0].columns.tolist()
    for c in ['round', 'seed', 'strategy']:
        metrics.remove(c)
        
    # get AUBC and last score
    metrics_results = defaultdict(list)
    
    for s, r in results.items():
        for m in metrics:
            metrics_results[f'AUBC {m.upper()}'].append(aubc(r.groupby(['round'])[m].mean().values))
            metrics_results[f'Last {m.upper()}'].append(r.groupby(['round'])[m].mean().values[-1])
            
    strategy_results = {'strategy' : list(results.keys())}
    strategy_results.update(metrics_results)
    table = pd.DataFrame(strategy_results)
        
    return table

def query_frequency(
    exp_names: list, savedir: str, model: str, 
    n_start: int, n_query: int, n_end: int, seed: int, change_names: dict = None, figsize: tuple = (10,3)
):
    """
    Query frequency per class
    
    Args:
    - strategy (str): strategy name for active learning
    - train_path (str): train dataset file path
    - savedir (str): saved result directory
    - model (str): model name
    - n_start (int): the number of initial samples for active learning
    - n_query (int): the number of query for active learning
    - n_end (int): the number of end samples for active learning
    - seed (int): seed number
    
    Output:
    - 1. bar-plot using frequency per class of intital dataset
    - 2. bar-plot using query frequency per class for active learning outcomes
    
    =============
    
    Example:
    
    seed = 0
    savedir = './results/SamsungAL'
    train_path = '/datasets/SamsungAL/train_seed42.csv'
    model = 'swin_base_patch4_window7_224.ms_in22k'
    n_start = 5000
    n_query = 20
    n_end = 6000
    
    # table is output from comparison_aubc function
    
    best_metric = 'AUBC F1'
    best_strategy = table.loc[table[best_metric].idxmax()].strategy
    
    query_frequency(
        strategy   = best_strategy,
        train_path = train_path,
        savedir    = savedir,
        model      = model,
        n_end      = n_end,
        n_query    = n_query,
        n_start    = n_start,
        seed       = seed
    )
    """
    
    # load saved config
    cfg_path = os.path.join(
        savedir, model, '*', exp_names[0], 
        f'total_{n_end}-init_{n_start}-query_{n_query}', 'seed0', 'configs.yaml'
    )
    cfg_path = glob(cfg_path)[0]
    cfg = OmegaConf.load(cfg_path)

    # load dataset
    if f"load_{cfg.DATASET.name.lower()}" in __import__('datasets').__dict__.keys():
        trainset, _ = __import__('datasets').__dict__[f"load_{cfg.DATASET.name.lower()}"](
            datadir  = cfg.DATASET.datadir, 
            img_size = cfg.DATASET.img_size,
            mean     = cfg.DATASET.mean, 
            std      = cfg.DATASET.std,
            aug_info = cfg.DATASET.aug_info,
            **cfg.DATASET.get('params', {})
        )
        labels = get_target_from_dataset(trainset)
        
    else:
        trainset = pd.read_csv(os.path.join(cfg.DATASET.datadir, cfg.DATASET.name, f'train_seed{cfg.DATASET.seed}.csv'))
        labels = trainset.label.values
    
    # load query logs
    query_results = {}
    for exp in exp_names:    
        p = os.path.join(
            savedir, model, '*', exp, 
            f'total_{n_end}-init_{n_start}-query_{n_query}', f'seed{seed}', 'query_log.csv'
        )
        
        query_path = glob(p)[0]

        # load query log
        df = pd.read_csv(query_path)
        df['label'] = labels

        # filtering NaN
        df = df[~df.query_round.isna()]

        # calculate class frequency per round
        df_round = df.groupby(['label','query_round']).idx.count().reset_index()

        # update
        if change_names is not None:
            if exp in change_names.keys():
                exp = change_names[exp]
                
        query_results[exp] = df_round
        
    fig, ax = plt.subplots(1, len(exp_names)+2, figsize=figsize)

    # trainset frequency
    label_id, label_cnt = np.unique(labels, return_counts=True)
    ax[0] = sns.barplot(
        x    = label_id,
        y    = label_cnt,
        ax   = ax[0]
    )
    ax[0].set_ylabel('Frequency')
    ax[0].set_xlabel('Class')
    ax[0].set_title('Train dataset')
    for container in ax[0].containers:
        ax[0].bar_label(container, fmt='%d', size=13)
        
    # init frequency
    for i, (name, q_r) in enumerate(query_results.items()):
        df_query = q_r[q_r.query_round=='round0'].groupby('label').idx.sum().reset_index()
        
        ax[1] = sns.barplot(
            x    = 'label',
            y    = 'idx',
            data = df_query,
            ax   = ax[1]
        )
        ax[1].set_ylabel('Frequency')
        ax[1].set_xlabel('Class')
        ax[1].set_title('Initial dataset')
        for container in ax[1].containers:
            ax[1].bar_label(container, fmt='%d', size=13)


    # query frequency
    for i, (name, q_r) in enumerate(query_results.items()):
        df_query = q_r[q_r.query_round!='round0'].groupby('label').idx.sum().reset_index()
        
        ax[i+2] = sns.barplot(
            x    = 'label',
            y    = 'idx',
            data = df_query,
            ax   = ax[i+2]
        )
        ax[i+2].set_ylabel('Frequency')
        ax[i+2].set_xlabel('Class')
        ax[i+2].set_title(name)
        for container in ax[i+2].containers:
            ax[i+2].bar_label(container, fmt='%d', size=13)
        
    plt.tight_layout()
    plt.show()
        
        
def comparison_per_class(
    savedir: str, strategy: str, seed_list: int, model: str,
    n_start: int, n_query: int, n_end: int, test: bool = True):
    """
    Comparison metrics of strategy per class using figure
    
    
    Args:
    - savedir (str): saved result directory
    - strategy (str): strategy for active learning
    - seed_list (list): seed list
    - model (str): model name
    - n_start (int): the number of initial samples for active learning
    - n_query (int): the number of query for active learning
    - n_end (int): the number of end samples for active learning
    - test (bool): use test results or not. if not, use best validation results
    
    Output:
    - line-plots using scores of metrics per round
    
    =============
    
    Example:
    
    seed_list = [0]
    savedir = './results/SamsungAL'
    model = 'swin_base_patch4_window7_224.ms_in22k'
    n_start = 5000
    n_query = 20
    n_end = 6000
    
    # table is output from comparison_aubc function
    
    best_metric = 'AUBC F1'
    best_strategy = table.loc[table[best_metric].idxmax()].strategy
    
    comparison_per_class(
        savedir   = savedir, 
        model     = model, 
        strategy  = best_strategy, 
        n_start   = n_start, 
        n_query   = n_query, 
        n_end     = n_end, 
        seed_list = seed_list
    )
    """
    
    # total round
    total_round = int((n_end - n_start) / n_query)

    df = pd.DataFrame()
    for seed in seed_list:
        # define filename
        al_filename = f'round{total_round}-seed{seed}'
        if not test:
            al_filename += '_best'
        al_filename += '-per_class.json'  

        # define file path
        f = os.path.join(
            savedir, model, strategy, 
            f'total_{n_end}-init_{n_start}-query_{n_query}', f'seed{seed}', f'round{total_round}-seed{seed}-per_class.json'
        )

        # get results
        result_cls = json.load(open(f, 'r'))

        df_seed = pd.DataFrame()

        for r, m in result_cls.items():    
            m_i = m.copy()
            del m_i['cm']

            df_i = pd.DataFrame(m_i)
            df_i['class'] = [f'{i}' for i in range(len(m_i))]
            df_i['round'] = int(r.strip('round'))

            df_seed = pd.concat([df_seed, df_i], axis=0)

        # append all seed results
        df = pd.concat([df, df_seed], axis=0)
        
    # figure
    metrics = ['f1','recall','precision','acc']

    row = 2
    col = 2
    fig, ax = plt.subplots(row,col,figsize=(10,7))

    for i in range(row*col):
        sns.lineplot(
            y    = metrics[i],
            x    = 'round',
            hue  = 'class',
            data = df,
            ax   = ax[i//col, i%col]
        )

        ax[i//col, i%col].set_xlabel('Round')
        ax[i//col, i%col].set_ylabel('Score')
        ax[i//col, i%col].set_title(metrics[i].upper())

        # legend
        if i == 0:
            lines, labels = ax[i//col, i%col].get_legend_handles_labels()

        ax[i//col, i%col].get_legend().remove()

    fig.legend(
        lines, labels, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.98), 
        frameon=False, fontsize=15, 
    )
    plt.tight_layout()
    plt.show()
    
    
def strategy_cumsum(
    exp_name: str, savedir: str, modelname: str, targets: list,
    n_end: int, n_start: int, n_query: int):
    # query log
    query_path = os.path.join(
        savedir, modelname, '*', exp_name, 
        f'total_{n_end}-init_{n_start}-query_{n_query}', 'seed0', 'query_log.csv'
    )
    query_path = glob(query_path)[0]
    
    query_log = pd.read_csv(query_path)
    query_log['targets'] = targets
    query_log_notna = query_log[~query_log['query_round'].isna()]
    query_log_gp = query_log_notna.groupby(['query_round','targets']).count().reset_index()
    query_log_gp['cumsum'] = query_log_gp.groupby(['targets'])['idx'].cumsum()
    
    return query_log_gp

def figure_query_cumsum(
    exp_names: list, modelname: str, savedir: str, 
    n_end: int, n_start: int, n_query: int, figsize: tuple, change_names: dict = None):
    
    # load saved config
    cfg_path = os.path.join(
        savedir, modelname, '*', exp_names[0], 
        f'total_{n_end}-init_{n_start}-query_{n_query}', 'seed0', 'configs.yaml'
    )
    cfg_path = glob(cfg_path)[0]
    cfg = OmegaConf.load(cfg_path)

    # load dataset
    trainset, _ = __import__('datasets').__dict__[f"load_{cfg.DATASET.name.lower()}"](
        datadir  = cfg.DATASET.datadir, 
        img_size = cfg.DATASET.img_size,
        mean     = cfg.DATASET.mean, 
        std      = cfg.DATASET.std,
        aug_info = cfg.DATASET.aug_info,
        **cfg.DATASET.get('params', {})
    )
    targets = get_target_from_dataset(trainset)
    
    # get cumsum results
    cumsum_results = []
    for exp in exp_names:
        r = strategy_cumsum(
            exp_name  = exp,
            savedir   = savedir,
            modelname = modelname,
            targets   = targets,
            n_end     = n_end,
            n_start   = n_start,
            n_query   = n_query
        )
        cumsum_results.append(r)

    # total round
    total_round = int((n_end - n_start) / n_query) + 1    
    
    fig, ax = plt.subplots(len(cumsum_results), total_round, figsize=figsize)

    y_max = max([r['cumsum'].max() for r in cumsum_results])
    for row, s in enumerate(cumsum_results):
        for round_i in range(total_round):
            sns.barplot(
                x    = 'targets',
                y    = 'cumsum',
                data = s[s.query_round == f'round{round_i}'],
                ax   = ax[row, round_i]
            )
            ax[row, round_i].set_ylim([0, y_max])
            ax[row, round_i].set_xlabel('Class')
            
            if round_i == 0:
                if change_names is not None:
                    if exp in change_names.keys():
                        exp = change_names[exp]
                        
                ylabel = f'{exp}\ncumsum'
            else:
                ylabel = 'cumsum'
            ax[row, round_i].set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()