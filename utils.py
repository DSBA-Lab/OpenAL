from _ctypes import PyObj_FromPtr  # see https://stackoverflow.com/a/15012814/355230
import json
import re
import os
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from collections import defaultdict
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError('Only lists and tuples can be wrapped')
        self.value = value

class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'  # Unique string pattern of NoIndent object ids.
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding NoIndent wrapped values.
        ignore = {'cls', 'indent'}

        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                    else super(MyEncoder, self).default(obj))

    def iterencode(self, obj, **kwargs):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.

        # Replace any marked-up NoIndent wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super(MyEncoder, self).iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_repr = json.dumps(no_indent.value, **self._kwargs)
                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                encoded = encoded.replace(
                            '"{}"'.format(format_spec.format(id)), json_repr)

            yield encoded
            
            
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
    savedir: str, strategies: list, seed_list: list, model: str, 
    n_start: int, n_query: int, n_end: int, test: bool = True, binary: bool = False) -> int and dict:
    """
    Extract active learning results
    
    Args:
    - savedir (str): saved result directory
    - strategies (list): strategies for active learning
    - seed_list (list): seed list
    - model (str): model name
    - n_start (int): the number of initial samples for active learning
    - n_query (int): the number of query for active learning
    - n_end (int): the number of end samples for active learning
    - test (bool): use test results or not. if not, use best validation results
    - binary (bool): use binary results of not
    
    Returns:
    - total_round (int): total round for active learning
    - r (dict): active learning results data frame by strategies
    
    =============
    
    Example:
    
    al_list = [
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
        savedir    = savedir,
        strategies = strategies,
        seed_list  = seed_list,
        model      = model,
        n_start    = n_start,
        n_query    = n_query,
        n_end      = n_end,
        test       = test,
        binary     = binary
    )
    """
    
    # total round
    total_round = int((n_end-n_start)/n_query)
    
    # get results
    r = defaultdict(list)
    for s in strategies:

        df_all = pd.DataFrame()

        # concat result dataframe for seed
        for seed in seed_list:
            
            # define filename
            al_filename = f'round{total_round}-seed{seed}'
            if not test:
                al_filename += '_best'
            
            if binary:
                al_filename += '-per_class.json'
            else:
                al_filename += '.csv'            
            
            # get result
            f = os.path.join(
                savedir, model, s,
                f'total_{n_end}-init_{n_start}-query_{n_query}', f'seed{seed}', al_filename
            )
            
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
        df_all['strategy'] = s
        r[s] = df_all
        
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
    r_full = defaultdict(list)
    
    # dataset seed
    for f in fullname_list:    
        # model seed
        for seed in seed_list:
            # define filename
            full_filename = f'results-seed{seed}'
            if not test:
                full_filename = full_filename.replace('-seed', '_seed')
                full_filename += '_best'
                
            if binary:
                full_filename += '-per_class.json'
            else:
                full_filename += '.json'   
            
            # get results
            full_s = json.load(open(os.path.join(savedir, model, f, full_filename), 'r'))
            
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
    
    return r_full


def comparison_strategy(
    savedir: str, strategies: list, fullname_list: list, seed_list: list, model: str, 
    n_start: int, n_query: int, n_end: int, 
    savepath: str = None, figsize: tuple = (7,10), test: bool = True, binary: bool = False) -> dict:
    """
    Comparison strategies results using figure
    
    Args:
    - savedir (str): saved result directory
    - strategies (list): strategies for active learning
    - fullname_list (list): full images supervised learning experiment foldernames
    - seed_list (list): seed list
    - model (str): model name
    - n_start (int): the number of initial samples for active learning
    - n_query (int): the number of query for active learning
    - n_end (int): the number of end samples for active learning
    - test (bool): use test results or not. if not, use best validation results
    - binary (bool): use binary results of not
    
    Return:
    - r (dict): active learning results data frame by strategies
    
    =============
    
    Example:
    
    al_list = [
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
        strategies    = al_list,
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
        savedir    = savedir,
        strategies = strategies,
        seed_list  = seed_list,
        model      = model,
        n_start    = n_start,
        n_query    = n_query,
        n_end      = n_end,
        test       = test,
        binary     = binary
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
    metrics = ['auroc','f1','recall','precision','bcr','acc']
    row = 2
    col = len(metrics)//row
    fig, ax = plt.subplots(row, col, figsize=figsize)

    for i in range(row*col):
        strategy_figure(
            data        = r,
            data_full   = r_full,
            metric      = metrics[i],
            total_round = total_round,
            n_query     = n_query,
            n_start     = n_start,
            ax          = ax[i//col, i%col]
        )
    
        if i == 0:
            lines, labels = ax[i//col, i%col].get_legend_handles_labels()
        
        ax[i//col, i%col].get_legend().remove()

    fig.legend(
        lines, labels, ncol=6, loc='lower center', bbox_to_anchor=(0.5, 0.98), 
        frameon=False, fontsize=15, 
    )
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()
            
        
    return r


def strategy_figure(data, data_full, metric, total_round, n_query, n_start, ax):
    data = pd.concat(list(data.values()))
    
    sns.lineplot(
        x    = 'round',
        y    = metric,
        hue  = 'strategy',
        data = data,
        ax   = ax
    )
    
    # full supervised learning
    if data_full != None:
        ax.axhline(y=data_full[metric].mean(), color='black')
        ax.axhline(y=data_full[metric].mean() + data_full[metric].std(), linestyle='--', color='black')
        ax.axhline(y=data_full[metric].mean() - data_full[metric].std(), linestyle='--', color='black')   
        
    # figure info
    ax.set_title(metric.upper())
    ax.set_ylabel('Score')
    ax.set_xlabel('The Number of Labeled Images')
    ax.set_xticks(
        (np.arange(0,total_round+1,5)).astype(int), 
        (np.arange(0,total_round+1,5)*n_query + n_start).astype(int), 
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
    strategy: str, train_path: str, savedir: str, model: str, 
    n_start: int, n_query: int, n_end: int, seed: int, figsize: tuple = (10,3)
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
    
    p = os.path.join(
        savedir, model, strategy, 
        f'total_{n_end}-init_{n_start}-query_{n_query}', f'seed{seed}', 'query_log.csv'
    )

    log = pd.read_csv(p)
    train = pd.read_csv(train_path)

    # concat log with train info
    df = pd.concat([train, log], axis=1)

    # filtering NaN
    df = df[~df.query_round.isna()]

    # calculate class frequency per round
    df_round = df.groupby(['label','query_round']).idx.count().reset_index()

    fig, ax = plt.subplots(1,2,figsize=figsize)

    # init frequency
    df_init = df_round[df_round.query_round=='round0']
    ax[0] = sns.barplot(
        x    = 'label',
        y    = 'idx',
        data = df_init,
        ax   = ax[0]
    )
    ax[0].set_ylabel('Frequency')
    ax[0].set_xlabel('Class')
    ax[0].set_title('Initial dataset')
    for container in ax[0].containers:
        ax[0].bar_label(container, fmt='%d', size=13)

    # query frequency
    df_query = pd.merge(
        pd.DataFrame({'label':df.label.unique()}),
        df_round[df_round.query_round!='round0'].groupby('label').idx.sum().reset_index(),
        on = 'label',
        how = 'left'
    )
    df_query = df_query.fillna(0)
    ax[1] = sns.barplot(
        x    = 'label',
        y    = 'idx',
        data = df_query,
        ax   = ax[1]
    )
    ax[1].set_ylabel('Frequency')
    ax[1].set_xlabel('Class')
    ax[1].set_title('Query')
    for container in ax[1].containers:
        ax[1].bar_label(container, fmt='%d', size=13)
        
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