import os
import json
import wandb
import pandas as pd
import numpy as np

from accelerate import Accelerator
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

import timm
from timm import create_model

from datasets import create_dataset
from query_strategies import torch_seed
from query_strategies import create_query_strategy, \
                             create_is_labeled_unlabeled, \
                             create_id_ood_targets, \
                             create_id_testloader, \
                             create_scheduler, create_optimizer
from query_strategies.utils import MyEncoder                             
from train import load_resume, test, fit
from arguments import parser
from main import make_directory

class ImageEncoder(nn.Module):
    def __init__(self, modelname, num_classes, img_size, only_classifier: bool = False):
        super(ImageEncoder, self).__init__()
        
        self.only_classifier = only_classifier
        self.encoder = create_model(modelname, pretrained=True, img_size=img_size, num_classes=0)
        for param in self.encoder.parameters():
            param.requires_grad = False
    
        self.num_classes = num_classes
        self.fc = nn.Linear(self.encoder.embed_dim, num_classes)
    
    @torch.no_grad()
    def forward_features(self, x):
        if self.only_classifier:
            return x
        else:
            out = self.encoder(x)
            return out
        
    def forward(self, x):
        out = self.forward_features(x)
        out = self.fc(out)
        
        return out

class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, num_per_cls: list):
        super(BalancedSoftmax, self).__init__()
        self.num_per_cls = torch.tensor(num_per_cls)

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(labels=label, logits=input, num_per_cls=self.num_per_cls, reduction=reduction)


def balanced_softmax_loss(labels, logits, num_per_cls, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      num_per_cls: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    npc = num_per_cls.type_as(logits)
    npc = npc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + npc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


# def create_is_labeled_unlabeled(trainset, size: int, seed: int):
#     '''
#     Args:
#     - trainset (torch.utils.data.Dataset): trainset
#     - size (int): represents the absolute number of train samples. 
#     - seed (int): seed for random state
    
#     Return:
#     - labeled_idx (np.ndarray): selected labeled indice
#     - unlabeled_idx (np.ndarray): selected unlabeled indice
#     '''
    
#     torch_seed(seed)

#     id_total_idx = trainset.file_info[trainset.file_info['domain'] == trainset.in_domain].index.tolist()
#     ood_total_idx = trainset.file_info[trainset.file_info['domain'] != trainset.in_domain].index.tolist()
    
#     print("# Total ID: {}, OOD: {}".format(len(id_total_idx), len(ood_total_idx)))

#     c_size = int(size / len(trainset.classes))

#     id_class_idx = trainset.file_info.loc[trainset.file_info['domain'] == trainset.in_domain, 'class_idx']

#     lb_idx = []
#     for c in id_class_idx.unique():
#         id_c_idx = np.random.choice(id_class_idx[id_class_idx == c].index, size=c_size, replace=False)
#         lb_idx.extend(id_c_idx)
        
#     ulb_idx = list(set(id_total_idx + ood_total_idx) - set(lb_idx))
#     print("# Labeled in: {}, Unlabeled: {}".format(len(lb_idx), len(ulb_idx)))

#     # defined empty labeled index
#     is_labeled = np.zeros(len(trainset), dtype=bool)
#     is_unlabeled = np.zeros(len(trainset), dtype=bool)
#     is_ood = np.zeros(len(trainset), dtype=bool)

#     is_labeled[lb_idx] = True
#     is_unlabeled[ulb_idx] = True

#     return is_labeled, is_unlabeled, is_ood



def openset_al_run(cfg: dict, trainset, testset, savedir: str):
    torch_seed(cfg.DEFAULT.seed)

    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
        mixed_precision             = cfg.TRAIN.mixed_precision
    )
    
    # set device
    print('Device: {}'.format(accelerator.device))

    # set active learning arguments
    nb_round = (cfg.AL.n_end - cfg.AL.n_start)/cfg.AL.n_query
    
    if nb_round % int(nb_round) != 0:
        nb_round = int(nb_round) + 1
    else:
        nb_round = int(nb_round)
    
    # logging
    print('[total samples] {}, [initial samples] {} [query samples] {} [end samples] {} [total round] {}'.format(
        len(trainset), cfg.AL.n_start, cfg.AL.n_query, cfg.AL.n_end, nb_round))
    
    # create ID and OOD targets
    trainset, id_targets = create_id_ood_targets(
        dataset      = trainset,
        nb_id_class  = cfg.AL.nb_id_class,
        seed         = cfg.DEFAULT.seed,
        id_targets   = cfg.DATASET.get('predefined_id_targets', []),
        use_majority = cfg.DATASET.get('use_majority', False)
    )
    testset, id_targets_check = create_id_ood_targets(
        dataset      = testset,
        nb_id_class  = cfg.AL.nb_id_class,
        seed         = cfg.DEFAULT.seed,
        id_targets   = id_targets,
    )
    assert sum(id_targets == id_targets_check) == cfg.AL.nb_id_class, "ID targets are not matched"
    
    # save selected ID targets
    json.dump(
        obj    = {'target_ids': list(map(int, id_targets))},
        fp     = open(os.path.join(savedir, 'target_ids.json'), 'w'), 
        indent = '\t'
    )
    
    # inital sampling labeling
    is_labeled, is_unlabeled, is_ood = create_is_labeled_unlabeled(
        trainset   = trainset,
        id_targets = id_targets,
        size       = cfg.AL.n_start,
        ood_ratio  = cfg.AL.ood_ratio,
        seed       = cfg.DEFAULT.seed,
        method     = cfg.AL.init.method,
        init_ood   = cfg.AL.init.get('init_ood', True),
    )

    # load visual encoder
    if cfg.TRAIN.get('params', False).get('return_features', False):
        only_classifier = True
        trainset.get_features(savedir=cfg.TRAIN.params.savedir_features, modelname=cfg.MODEL.name, is_train=True)
        testset.get_features(savedir=cfg.TRAIN.params.savedir_features, modelname=cfg.MODEL.name, is_train=False)
        trainset.return_features = True
        testset.return_features = True
    else:
        only_classifier = False
        
    model = ImageEncoder(
        modelname       = cfg.MODEL.name,
        num_classes     = cfg.DATASET.num_classes,
        img_size        = cfg.DATASET.img_size,
        only_classifier = only_classifier
    )
    data_config = timm.data.resolve_model_data_config(model.encoder)
    data_config['input_size'] = (3, cfg.DATASET.img_size, cfg.DATASET.img_size)
    trainset.transform = timm.data.create_transform(**data_config, is_training=True)
    testset.transform = timm.data.create_transform(**data_config, is_training=False)

    # select strategy    
    openset_params = {
        'is_openset'      : True,
        'is_unlabeled'    : is_unlabeled,
        'is_ood'          : is_ood,
        'id_classes'      : trainset.classes[id_targets],
        'savedir'         : savedir,
        'seed'            : cfg.DEFAULT.seed,
        'accelerator'     : accelerator
    }
    openset_params.update(cfg.AL.get('openset_params', {}))
    openset_params.update(cfg.AL.get('params', {}))
    
    strategy = create_query_strategy(
        strategy_name    = cfg.AL.strategy, 
        model            = model,
        dataset          = trainset, 
        transform        = testset.transform,
        sampler_name     = cfg.DATASET.sampler_name,
        is_labeled       = is_labeled, 
        n_query          = cfg.AL.n_query, 
        n_subset         = cfg.AL.n_subset,
        batch_size       = cfg.DATASET.test_batch_size, 
        num_workers      = cfg.DATASET.num_workers,
        steps_per_epoch  = cfg.TRAIN.params.get('steps_per_epoch', 0),
        use_diverse      = cfg.AL.get('use_diverse', False),
        **openset_params
    )
    
    # define test dataloader
    testloader = create_id_testloader(
        dataset     = testset,
        id_targets  = id_targets,
        batch_size  = cfg.DATASET.test_batch_size,
        num_workers = cfg.DATASET.num_workers    
    )
    
    # resume
    if cfg.TRAIN.get('resume', False).get('use', False):
        is_resume = True
        
        resume_info = load_resume(
            ckp_path   = cfg.TRAIN.resume.ckp_path,
            ckp_round  = cfg.TRAIN.resume.ckp_round,
            strategy   = strategy,
            seed       = cfg.DEFAULT.seed,
            is_openset = cfg.AL.get('ood_ratio', False)
        )
        
        model = resume_info['model']
        strategy = resume_info['strategy']
        start_round = resume_info['start_round']
        history = resume_info['history']
        
        query_log_df, nb_labeled_df = history['query_log'], history['nb_labeled']
        log_df_test = history['log']['test']
        metrics_log_test = history['metrics']['test']
        
        model = accelerator.prepare(model)
        
    else:
        is_resume = False
        start_round = 0 
        
        # define log dataframe
        log_df_test = pd.DataFrame(
            columns=['round', 'auroc', 'f1', 'recall', 'precision', 'bcr', 'acc', 'loss']
        )
        
        # query log dataframe
        query_log_df = pd.DataFrame({'idx': range(len(is_labeled)), 'is_unlabel': np.zeros(len(is_labeled), dtype=bool)})
        query_log_df.loc[is_unlabeled, 'is_unlabel'] = True
        query_log_df['query_round'] = None
        query_log_df['ID_query_round'] = None
        query_log_df.loc[np.r_[np.where(is_labeled==True)[0], np.where(is_ood==True)[0]], 'query_round'] = 'round0'
        query_log_df.loc[is_labeled, 'ID_query_round'] = 'round0'
        
        # number of labeled set log dataframe
        nb_labeled_df = pd.DataFrame({'round': [0], 'nb_labeled': [is_labeled.sum()]})
        
        # define dict to save confusion matrix and metrics per class
        metrics_log_test = {}
        
    nb_round += start_round
    
    # run
    for r in range(start_round, nb_round+1):
        
        if r != 0:    
            # query sampling    
            query_idx = strategy.query(model)
            
            # update query
            id_query_idx = strategy.update(query_idx=query_idx)
            
            # check resume
            if is_resume:
                is_resume = False
            
            # define accelerator
            accelerator = Accelerator(
                gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
                mixed_precision             = cfg.TRAIN.mixed_precision
            )
            if hasattr(strategy, 'accelerator'):
                strategy.accelerator = accelerator
            
            # save query index
            query_log_df.loc[query_idx, 'query_round'] = f'round{r}'
            query_log_df.loc[id_query_idx, 'ID_query_round'] = f'round{r}'
            query_log_df.to_csv(os.path.join(savedir, 'query_log.csv'), index=False)
            
            # save nb_labeled
            nb_labeled_r = {'round': [r], 'nb_labeled': [strategy.is_labeled.sum()]}
            nb_labeled_df = pd.concat([nb_labeled_df, pd.DataFrame(nb_labeled_r, index=[len(nb_labeled_df)])], axis=0)
            nb_labeled_df.to_csv(os.path.join(savedir, 'nb_labeled.csv'), index=False)
            
        # logging
        print('[Round {}/{}] training samples: {}'.format(r, nb_round, sum(strategy.is_labeled)))
        
        # build Model          
        if not cfg.AL.get('continual', False) or r == 0:
            model = strategy.init_model()
            model = accelerator.prepare(model)
        
        # get trainloader
        trainloader = strategy.get_trainloader()
        trainloader, testloader = accelerator.prepare(trainloader, testloader)
        
        # optimizer
        optimizer = create_optimizer(
            opt_name   = cfg.OPTIMIZER.name, 
            model      = model, 
            lr         = cfg.OPTIMIZER.lr, 
            opt_params = cfg.OPTIMIZER.get('params',{}),
            backbone   = True
        )

        # scheduler
        scheduler = create_scheduler(
            sched_name    = cfg.SCHEDULER.name, 
            optimizer     = optimizer, 
            epochs        = cfg.TRAIN.epochs, 
            params        = cfg.SCHEDULER.get('params', {}),
            warmup_params = cfg.SCHEDULER.get('warmup_params', {})
        )
        for k, opt in optimizer.items():
            optimizer[k] = accelerator.prepare(opt)
            
        for k, sched in scheduler.items():
            scheduler[k] = accelerator.prepare(sched)

        # initialize wandb
        if cfg.TRAIN.wandb.use:
            wandb.init(name=f'{cfg.DEFAULT.exp_name}_round{r}', project=cfg.TRAIN.wandb.project_name, entity=cfg.TRAIN.wandb.entity, config=OmegaConf.to_container(cfg))


        # strategy.criterion = BalancedSoftmax(num_per_cls=np.unique(trainset.targets[strategy.is_labeled], return_counts=True)[1])
        # fitting model
        fit(
            model        = model, 
            trainloader  = trainloader, 
            testloader   = None, 
            criterion    = strategy.loss_fn, 
            optimizer    = optimizer, 
            scheduler    = scheduler,
            accelerator  = accelerator,
            epochs       = cfg.TRAIN.epochs, 
            use_wandb    = cfg.TRAIN.wandb.use,
            log_interval = cfg.TRAIN.log_interval,
            seed         = cfg.DEFAULT.seed,
            **cfg.TRAIN.get('params', {})
        )

        # ====================
        # test results
        # ====================
        
        test_results = test(
            model            = model, 
            dataloader       = testloader, 
            criterion        = strategy.loss_fn, 
            log_interval     = cfg.TRAIN.log_interval,
            return_per_class = True,
            **cfg.TRAIN.get('params', {})
        )

        # save results per class
        metrics_log_test.update({
            f'round{r}': test_results['per_class']
        })
        json.dump(
            obj    = metrics_log_test, 
            fp     = open(os.path.join(savedir, f"round{nb_round}-seed{cfg.DEFAULT.seed}_test-per_class.json"), 'w'), 
            cls    = MyEncoder,
            indent = '\t'
        )
        
        del test_results['per_class']
        
        # save results 
        log_metrics = {'round':r}
        log_metrics.update([(k, v) for k, v in test_results.items()])
        log_df_test = pd.concat([log_df_test, pd.DataFrame(log_metrics, index=[len(log_df_test)])], axis=0)
        
        log_df_test.round(4).to_csv(
            os.path.join(savedir, f"round{nb_round}-seed{cfg.DEFAULT.seed}_test.csv"),
            index=False
        )   
        
        print('append result [shape: {}]'.format(log_df_test.shape))
        
        wandb.finish()
        
def run(cfg):
    trainset, _, testset = create_dataset(
        datadir  = cfg.DATASET.datadir, 
        dataname = cfg.DATASET.name,
        img_size = cfg.DATASET.img_size,
        mean     = cfg.DATASET.mean,
        std      = cfg.DATASET.std,
        aug_info = cfg.DATASET.aug_info,
        **cfg.DATASET.get('params', {})
    )
    
    # make save directory
    al_name = f"total_{cfg.AL.n_end}-init_{cfg.AL.n_start}-query_{cfg.AL.n_query}"
    savedir = os.path.join(
        cfg.DEFAULT.savedir, cfg.DATASET.name, cfg.MODEL.name, 
        cfg.AL.strategy, cfg.DEFAULT.exp_name, al_name, f'seed{cfg.DEFAULT.seed}'
    )
    
    make_directory(
        savedir   = savedir,
        is_resume = cfg.TRAIN.get('resume', False).get('use', False)
    )
    
    # save config
    OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
    
    openset_al_run(
        cfg         = cfg,
        trainset    = trainset,
        testset     = testset,
        savedir     = savedir,
    )    
        
        
if __name__=='__main__':
    # config
    cfg = parser()
    
    # run
    run(cfg)