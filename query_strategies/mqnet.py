import os
import numpy as np
from PIL import Image
from copy import deepcopy
from tqdm.auto import tqdm
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from metric_learning.models import MetricModel
from query_strategies.learning_loss import LearningLoss
from query_strategies.scheds import create_scheduler
from query_strategies.optims import create_optimizer
from query_strategies.factory import create_query_strategy
from query_strategies.strategy import Strategy
from query_strategies.utils import torch_seed, get_target_from_dataset

class MQNet(Strategy):
    def __init__(
        self,
        seed: int, 
        savedir: str, 
        selected_strategy: str, 
        meta_params: dict = {}, 
        metric_params: dict= {},
        accelerator = None,
        **init_args
    ):
        
        super(MQNet, self).__init__(**init_args)
        
        # make query strategy
        del_keys = ['is_openset', 'is_unlabeled', 'is_ood', 'id_classes']
        for k in del_keys:
            del init_args[k]
        
        self.query_strategy = create_query_strategy(strategy_name=selected_strategy, **init_args)
        assert 'get_scores' in dir(self.query_strategy), f'{selected_strategy} is not able to obtain an informativeness score.'
        
        self.savedir = savedir
                
        self.accelerator = accelerator
            
        # meta learning
        self.meta_learning = MetaLearning(
            num_id_class    = self.num_id_class,
            savedir         = self.savedir, 
            dim             = meta_params['dim'],
            epochs          = meta_params['epochs'],
            steps_per_epoch = meta_params['steps_per_epoch'],
            batch_size      = meta_params['batch_size'],
            num_workers     = meta_params['num_workers'],
            margin          = meta_params['margin'],
            opt_name        = meta_params['opt_name'],
            opt_params      = meta_params.get('opt_params', {}),
            lr              = meta_params['lr'],
            sched_name      = 'multi_step',
            sched_params    = {'milestones': [int(meta_params['epochs']/2)]},
            warmup_params   = meta_params.get('warmup_params', {}),
            accelerator     = accelerator,
            seed            = seed, 
        )
        
        self.vis_encoder = MetricModel(
            modelname   = metric_params['modelname'], 
            pretrained  = metric_params['pretrained'],
            simclr_dim  = metric_params['simclr_dim'],
            **metric_params.get('model_params', {})
        )
        self.checkpoint_path = os.path.join(
            metric_params['checkpoint_path'], 
            self.dataset.dataname, 
            'SimCLRCSI',
            f"{metric_params['modelname']}",
            "ckp.pt"
        )

    def init_ssl_model(self, device):
        vis_encoder = deepcopy(self.vis_encoder)
        vis_encoder.load_state_dict(torch.load(self.checkpoint_path))
        
        if self.accelerator != None:
            vis_encoder = self.accelerator.prepare(vis_encoder)
        else:
            vis_encoder.to(device)
            
        vis_encoder.eval()
        
        return vis_encoder
        
    def init_model(self):
        return self.query_strategy.init_model()
    
    def loss_fn(self, outputs, targets):
        return self.query_strategy.loss_fn(outputs=outputs, targets=targets)
        
    
    def query(self, model, **kwargs) -> np.ndarray:
        # device
        device = next(model.parameters()).device
        
        # idx
        labeled_idx = np.where(self.is_labeled==True)[0]
        unlabeled_idx = np.where(self.is_unlabeled==True)[0]
        
        # get purity scores for unlabeled samples
        purity_scores = self.get_purity_score(
            unlabeled_idx = unlabeled_idx, 
            labeled_idx   = labeled_idx, 
            device        = device
        )
        purity_scores = self.standardize(scores=purity_scores)
        
        # get informativeness scores for unlabeled samples
        informativeness_scores, _ = self.query_strategy.get_scores(model=model, sample_idx=unlabeled_idx)
        informativeness_scores = self.standardize(scores=informativeness_scores)
        
        # create meta input
        meta_inputs = self.create_meta_input(informativeness=informativeness_scores, purity=purity_scores)
            
        # get meta scores
        meta_scores = self.get_meta_scores(meta_inputs=meta_inputs, device=device)
            
        # select query
        score_rank = meta_scores.sort(descending=True)[1]
        query_rank = score_rank[:self.n_query]
        select_idx = unlabeled_idx[query_rank]
        
        # meta-learning for next round
        self.meta_fit(
            model                  = model,
            labeled_idx            = labeled_idx, 
            select_idx             = select_idx, 
            informativeness_scores = informativeness_scores[query_rank],
            device                 = device
        )
        
        return select_idx
    
    def meta_fit(self, model, labeled_idx: np.ndarray, select_idx: np.ndarray, informativeness_scores: torch.Tensor, device: str):
        targets = get_target_from_dataset(self.dataset)
        
        purity_scores = self.get_purity_score(
            unlabeled_idx = select_idx, 
            labeled_idx   = labeled_idx, 
            device        = device
        )
        purity_scores = self.standardize(scores=purity_scores)
        
        inputs = torch.stack([informativeness_scores, purity_scores], dim=1)
        self.meta_learning.fit(
            model     = model,
            X         = self.dataset.data[select_idx],
            X_meta    = inputs,
            y         = targets[select_idx],
            transform = self.dataset.transform,
            device    = device
        )
    
    
    def get_meta_scores(self, meta_inputs: torch.FloatTensor, device: str):
        if self.meta_learning.current_round == 0:
            meta_scores = meta_inputs
        else:
            mqnet = self.meta_learning.init_model(device=device, load_model=True)
            with torch.no_grad():
                meta_scores = mqnet(meta_inputs.to(device)).squeeze()
    
        return meta_scores.cpu()
    
    def create_meta_input(self, informativeness: torch.FloatTensor, purity: torch.FloatTensor):
        # standardize
        if self.meta_learning.current_round == 0:
            meta_input = informativeness + purity
            return meta_input
        else:
            # concat
            meta_input = torch.stack([informativeness, purity], dim=1)
            
            return meta_input
    
    def standardize(self, scores: torch.FloatTensor, ref_scores: torch.FloatTensor = None):
        if ref_scores != None:
            std, mean = torch.std_mean(ref_scores, unbiased=False)
        else:
            std, mean = torch.std_mean(scores, unbiased=False)
            
        scores = (scores - mean) / std
        scores = torch.exp(scores)
        
        return scores
    
    
    def get_purity_score(self, unlabeled_idx: np.ndarray, labeled_idx: np.ndarray, device: str):
        # fit representation model using CSI
        vis_encoder = self.init_ssl_model(device=device)
        
        # get labeled and unlabeled normalized features using representation model
        ulb_normalized_embed = self.extract_outputs(
            model        = vis_encoder, 
            sample_idx   = unlabeled_idx, 
            return_probs = False,
            return_embed = True
        )['embed']
        
        lb_normalized_embed = self.extract_outputs(
            model        = vis_encoder, 
            sample_idx   = labeled_idx, 
            return_probs = False,
            return_embed = True
        )['embed']
        
        purity_scores = torch.matmul(ulb_normalized_embed, lb_normalized_embed.t())
        purity_scores = purity_scores.max(dim=1)[0]

        return purity_scores
    
    
    
class QueryNet(nn.Module):
    def __init__(self, inter_dim: int = 64):
        super().__init__()

        input_size = 2 # purity and informativeness scores

        w1 = torch.rand(input_size, inter_dim, requires_grad=True) #ones
        w2 = torch.rand(inter_dim, 1, requires_grad=True) #ones
        b1 = torch.rand(inter_dim, requires_grad=True) #zeros
        b2 = torch.rand(1, requires_grad=True) #zeros

        self.w1 = torch.nn.parameter.Parameter(w1, requires_grad=True)
        self.w2 = torch.nn.parameter.Parameter(w2, requires_grad=True)
        self.b1 = torch.nn.parameter.Parameter(b1, requires_grad=True)
        self.b2 = torch.nn.parameter.Parameter(b2, requires_grad=True)


    def forward(self, x):
        out = torch.sigmoid(torch.matmul(x, torch.relu(self.w1)) + self.b1)
        out = torch.matmul(out, torch.relu(self.w2)) + self.b2
        return out


class MetaLearning:
    def __init__(
        self, 
        num_id_class: int,
        savedir: str,
        dim: int = 64,
        epochs: int = 100,
        steps_per_epoch: int = 100,
        batch_size: int = 32,
        num_workers: int = 4,
        margin: float = 1.,
        opt_name: str = 'SGD',
        lr: float = 0.001,
        seed: int = 223, 
        opt_params: dict = {}, 
        sched_name: str = 'multi_step',
        sched_params: dict = {},
        warmup_params: dict = {},
        accelerator = None,
        **kwargs
    ):
        
        self.accelerator = accelerator
        
        self.mqnet = QueryNet(inter_dim=dim)
        self.num_id_class = num_id_class
        
        # training
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # loss functions
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.learningloss = LearningLoss(margin=margin)
        
        # optimizer
        self.opt_name = opt_name
        self.lr = lr
        self.opt_params = opt_params
        
        # scheduler
        self.sched_name = sched_name
        self.sched_params = sched_params
        self.warmup_params = warmup_params
        
        self.seed = seed
        
        # save
        self.savedir = savedir
        self.current_round = len(glob(os.path.join(self.savedir, 'meta_model*')))
        
        
    def init_model(self, device: str, load_model: bool = False):
        mqnet = deepcopy(self.mqnet)
        
        if load_model:
            mqnet.load_state_dict(torch.load(os.path.join(self.savedir, f'meta_model{self.current_round}.pt')))
            mqnet.eval()
        
        mqnet.to(device)
            
        return mqnet

        
    def fit(self, model, X: torch.FloatTensor, X_meta: torch.FloatTensor, y: torch.LongTensor, transform, device: str, **kwargs):
        torch_seed(self.seed)
        
        # create MQNet
        mqnet = self.init_model(device=device)
        
        # split dataset
        self.create_trainset(X=X, X_meta=X_meta, y=y, transform=transform)

        # optimizer
        optimizer = create_optimizer(opt_name=self.opt_name, model=mqnet, lr=self.lr, opt_params=self.opt_params)
        scheduler = create_scheduler(
            sched_name    = self.sched_name, 
            optimizer     = optimizer, 
            epochs        = self.epochs, 
            params        = self.sched_params,
            warmup_params = self.warmup_params
        )
        
        # accelerator
        if self.accelerator != None:
            mqnet, optimizer, scheduler, self.trainloader = self.accelerator.prepare(mqnet, optimizer, scheduler, self.trainloader)

        # training
        desc = '[Meta-Query Net] lr: {lr:.3e}'
        p_bar = tqdm(range(self.epochs), total=self.epochs)
        model.eval()
        for _ in p_bar:
            p_bar.set_description(desc=desc.format(lr=optimizer.param_groups[0]['lr']))
            
            self.train(model=model, mqnet=mqnet, optimizer=optimizer, device=device)
            scheduler.step()
            
        self.save_model(mqnet=mqnet)
        
    def save_model(self, mqnet):
        # start meta-learning start in second round
        torch.save(mqnet.state_dict(), os.path.join(self.savedir, f'meta_model{self.current_round+1}.pt'))
        self.current_round += 1
            
        
    def create_trainset(self, X: torch.FloatTensor, X_meta: torch.FloatTensor, y: torch.LongTensor, transform):
        if not isinstance(y, torch.Tensor):
            y = torch.LongTensor(y)
            
        dataset = MetaDataset(X=X, X_meta=X_meta, y=y, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        
        setattr(self, 'trainloader', dataloader)
    
    def train(self, model, mqnet, optimizer, device: str):
        total_loss = 0
        
        desc = '[TRAIN] Loss: {loss:>6.4f}'
        p_bar = tqdm(total=self.steps_per_epoch, desc=desc.format(loss=total_loss), leave=False)
              
        mqnet.train()
        
        batch_idx = 0
        
        while batch_idx < self.steps_per_epoch:
            for idx, (inputs, inputs_meta, targets) in enumerate(self.trainloader):   
                if self.accelerator == None:
                    inputs, inputs_meta, targets = inputs.to(device), inputs_meta.to(device), targets.to(device)

                is_id = targets.le(self.num_id_class-1).type(torch.LongTensor).to(device)

                # get pred_scores through MQNet
                outputs_meta = mqnet(inputs_meta).squeeze()

                # get outputs
                with torch.no_grad():
                    backbone = getattr(model, 'backbone', model)
                    outputs = backbone(inputs)
                    
                # get masked cross-entropy loss
                masked_targets = targets * is_id 
                masked_loss = self.criterion(outputs, masked_targets) * is_id

                # get learning loss
                loss = self.learningloss(outputs=outputs_meta, targets=masked_loss).mean()
                total_loss += loss.item()
                
                optimizer.zero_grad()
                if self.accelerator != None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                
                optimizer.step()
                
                batch_idx += 1
                    
                p_bar.update(1)
                p_bar.set_description(desc=desc.format(loss=total_loss/batch_idx))
                
                if batch_idx == self.steps_per_epoch:
                    break
                
        p_bar.close()
                    
                
class MetaDataset(Dataset):
    def __init__(self, X, X_meta, y, transform):
        self.X = X
        self.X_meta = X_meta
        self.y = y
        
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        img = self.X[i]
        
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        else:
            img = Image.fromarray(img)
            
        x_i = self.transform(img)
        x_meta_i = self.X_meta[i]
        y_i = self.y[i]
        
        return x_i, x_meta_i, y_i
        