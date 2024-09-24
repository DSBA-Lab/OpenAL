import json
import os
import wandb

import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator

import torch
import torch.nn as nn

from arguments import parser
from datasets import create_dataset
from query_strategies.open_clipn import load_model
from query_strategies.utils import MyEncoder
from query_strategies import create_query_strategy, \
                             create_id_ood_targets, create_is_labeled_unlabeled, \
                             create_id_testloader, \
                             create_scheduler, create_optimizer
from train import load_resume, fit, test
from main import make_directory


class ImageEncoder(nn.Module):
    def __init__(self, encoder, num_classes):
        super(ImageEncoder, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
        self.num_classes = num_classes
        self.fc = nn.Linear(self.encoder.output_dim, num_classes)
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        
        return out


def zeroshot_run(cfg, trainset, testset, savedir: str):
    # create ID and OOD targets
    trainset, id_targets = create_id_ood_targets(
        dataset     = trainset,
        nb_id_class = cfg.AL.nb_id_class,
        seed        = cfg.DEFAULT.seed
    )

    testset, id_targets_check = create_id_ood_targets(
        dataset     = testset,
        nb_id_class = cfg.AL.nb_id_class,
        seed        = cfg.DEFAULT.seed
    )
    
    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
        mixed_precision             = cfg.TRAIN.mixed_precision
    )
    
    # set device
    print('Device: {}'.format(accelerator.device))
    
    
    # load visual classifier
    vis_clf, _, process_test = load_model(
        model_type  = cfg.AL.openset_params.model_type, 
        pre_train   = cfg.AL.openset_params.ckp_path, 
        prompt_path = cfg.AL.openset_params.prompt_path, 
        classes     = testset.classes[id_targets]
    )
    
    vis_clf = vis_clf.cuda()
    vis_clf.eval()
    print('load model')
    
    testset.transform = process_test
    testloader = create_id_testloader(
        dataset     = testset,
        id_targets  = id_targets,
        batch_size  = cfg.DATASET.test_batch_size,
        num_workers = cfg.DATASET.num_workers    
    )

    with torch.no_grad():
        y_pred = []
        y_true = []
        for inputs, targets in tqdm(testloader, total=len(testloader)):
            logits, _, _ = vis_clf(inputs.cuda())
            
            y_pred.append(logits.argmax(dim=1).cpu())
            y_true.append(targets.cpu())
            
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    
    zeroshot_acc = y_true.eq(y_pred).sum() / len(y_true)
    
    json.dump({'top1 acc': zeroshot_acc.item()}, open(os.path.join(savedir, 'zeroshot_acc.json'), 'w'))
    
def openset_clipnal_run(cfg: dict, trainset, testset, savedir: str):

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
    print('[total samples] {}, [initial samples] {} [query samples] {} [end samples] {} [total round] {} [OOD ratio] {}'.format(
        len(trainset), cfg.AL.n_start, cfg.AL.n_query, cfg.AL.n_end, nb_round, cfg.AL.ood_ratio))
    
    # create ID and OOD targets
    trainset, id_targets = create_id_ood_targets(
        dataset     = trainset,
        nb_id_class = cfg.AL.nb_id_class,
        seed        = cfg.DEFAULT.seed
    )
    testset, id_targets_check = create_id_ood_targets(
        dataset     = testset,
        nb_id_class = cfg.AL.nb_id_class,
        seed        = cfg.DEFAULT.seed
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
        seed       = cfg.DEFAULT.seed
    )
    
    # load visual classifier
    model, process_train, process_test = load_model(
        model_type  = cfg.AL.openset_params.model_type, 
        pre_train   = cfg.AL.openset_params.ckp_path, 
        prompt_path = cfg.AL.openset_params.prompt_path, 
        classes     = testset.classes[id_targets]
    ) 
    if not cfg.DEFAULT.zeroshot:
        model = ImageEncoder(encoder=model.image_encoder, num_classes=cfg.DATASET.num_classes)
        
    trainset.transform = process_train
    testset.transform = process_test

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
        batch_size       = cfg.DATASET.batch_size, 
        num_workers      = cfg.DATASET.num_workers,
        steps_per_epoch  = cfg.TRAIN.params.get('steps_per_epoch', 0),
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
            return_per_class = True
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
    
    if cfg.DEFAULT.zeroshot:
        savedir = os.path.join(cfg.DEFAULT.savedir, cfg.DATASET.name, cfg.MODEL.name, 'zeroshot', cfg.DEFAULT.exp_name, f'seed{cfg.DEFAULT.seed}')
        make_directory(
            savedir   = savedir,
            is_resume = cfg.TRAIN.get('resume', False).get('use', False)
        )
        # save config
        OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
        
        zeroshot_run(cfg=cfg, trainset=trainset, testset=testset, savedir=savedir)
    else:
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
        
        openset_clipnal_run(cfg=cfg, trainset=trainset, testset=testset, savedir=savedir)
        
if __name__=='__main__':
    # config
    cfg = parser()
    
    # run
    run(cfg)