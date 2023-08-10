import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from datasets import load_cifar10, create_dataset, stats
from models import create_model
from datasets import RotationDataset

import torch
import torch.nn as nn
import torch.optim as optim
from ssl_pretext.utils import progress_bar
import torch.backends.cudnn as cudnn

import numpy as np
from main import torch_seed
import argparse
from omegaconf import OmegaConf


rotation_best_acc = 0  # best test accuracy

def rotation_train(model, trainloader, testloader, criterion, optimizer, scheduler, save_path: str, epoch: int):

    ## Training
    for epoch in range(cfg['epoch']):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        # inputs: list, targets: list
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = [x.to(device) for x in inputs]
            targets = [y.to(device) for y in targets]
            
            # Inference      
            outputs = [model(x) for x in inputs]
        
            # Calculate loss
            loss = [criterion(x,y) for x,y in zip(outputs, targets)]
            loss = sum(loss)/4.
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Train loss
            train_loss += loss.item()
            
            # Output index
            predicteds = [y.argmax(1) for y in outputs]
            
            # Loss and Accuracy log
            total += targets[0].size(0)*4
            corrects = [p.eq(t).sum().item() for p,t in zip(predicteds, targets)]
            corrects = sum(corrects)
            correct += corrects
            
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    ## Test
    for epoch in range(cfg['epoch']):
        global rotation_best_acc
        model.eval()
        test_loss, correct, total = 0, 0, 0
    
        with torch.no_grad():
            # inputs: list, targets: list
            for batch_idx, (inputs, targets, img_idx) in enumerate(testloader):
                inputs = [x.to(device) for x in inputs]
                targets = [y.to(device) for y in targets]
                
                # Inference
                outputs = [model(x) for x in inputs]
                
                # Calculate loss
                loss = [criterion(x,y) for x,y in zip(outputs, targets)]
                loss = sum(loss)/4.
                test_loss += loss.item()
                
                # Output index
                predicteds = [y.argmax(1) for y in outputs]
                
                # Loss and Accuracy log
                total += targets[0].size(0)*4    
                corrects = [p.eq(t).sum().item() for p,t in zip(predicteds, targets)]
                corrects = sum(corrects)  
                correct += corrects

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
        # Save checkpoint.   
        if not os.path.isdir(save_path): 
            os.mkdir(save_path)
        
        acc = 100.*correct/total
        with open((os.path.join(save_path, 'best_rotation')),'a') as f: 
            f.write(str(acc)+':'+str(epoch)+'\n')
        if acc > rotation_best_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir((os.path.join(save_path, 'checkpoint'))):
                os.mkdir((os.path.join(save_path, 'checkpoint'))) 
            # save rotation weights
            torch.save(state, (os.path.join(save_path, 'checkpoint', 'rotation.pth'))) 
            rotation_best_acc = acc

    scheduler.step()
    
    return 1


def make_batch(model, testloader, criterion, save_path: str):
    
    ## load checkpoint
    checkpoint = torch.load((os.path.join(save_path, 'checkpoint', 'rotation.pth'))) 
    model.load_state_dict(checkpoint['model'])
    
    ## inference
    model.eval()
    test_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, img_idx) in enumerate(testloader):
            inputs = [x.to(device) for x in inputs]
            targets = [y.to(device) for y in targets]
            
            # Inference
            outputs = [model(x) for x in inputs]
        
            # Calculate loss
            loss = [criterion(x,y) for x,y in zip(outputs, targets)]
            loss = sum(loss)/4.
            test_loss += loss.item()
            
            # Acquire data loss
            predicted = outputs[0].argmax(1)
            total += targets[0].size(0)
            correct += predicted.eq(targets[0]).sum().item()

            loss = loss.item()
            s = str(float(loss)) + '_' + str(img_idx) + "\n"

            # Save data loss
            with open((os.path.join(save_path, 'rotation_loss.txt')), 'a') as f: 
                f.write(s)
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
    # Load data loss           
    with open((os.path.join(save_path, 'rotation_loss.txt')), 'r') as f: 
        losses = f.readlines()

    loss_1, name_2 = [], []

    for j in losses:
        loss_1.append(j[:-1].split('_')[0])
        name_2.append(j[:-1].split('_')[1])

    # sort loss
    sort_loss = np.array(loss_1)
    sort_index = np.argsort(sort_loss)
    x = sort_index.tolist()
    x.reverse()
    sort_index = np.array(x) # convert to high loss first

    # save index loss
    if not os.path.isdir((os.path.join(save_path, 'batch'))): 
        os.mkdir((os.path.join(save_path, 'batch'))) 
    
    for index in sort_index:
        with open(os.path.join(save_path, 'batch', f'batch_loss.txt'), 'a') as f:
            f.write(str(index)+'\n')
            
        
if __name__ == "__main__":
    
    ## define argument
    parser = argparse.ArgumentParser(description='Active Learning - SSL Pretext')
    parser.add_argument('--ssl_setting', type=str, default=None, help='ssl config file')
    
    args = parser.parse_args()
    
    ## load ssl config
    cfg = OmegaConf.load(args.ssl_setting)
    
    ## define seed
    torch_seed(cfg['seed'])
    
    ## load dataset statistical
    data_stat = stats.datasets[cfg['dataname']]

    ## load dataset
    if cfg['dataname'] == 'CIFAR10':
        trainset, testset = load_cifar10(
                                datadir = cfg['datadir'], 
                                img_size = data_stat['img_size'], 
                                mean = data_stat['mean'], 
                                std = data_stat['std'], 
                                aug_info = None
                             )
    elif cfg['dataname'] == 'SamsungAL':
        trainset, validset, testset = create_dataset(
                                        datadir = cfg['datadir'], 
                                        dataname = cfg['dataname'], 
                                        img_size = data_stat['img_size'], 
                                        mean = data_stat['mean'], 
                                        std = data_stat['std'], 
                                        aug_info = None
                                    )
        
    ## load rotation dataset
    Rotation_trainset = RotationDataset(dataset=trainset, is_train=True)
    Rotation_testset = RotationDataset(dataset=trainset, is_train=False)
    
    Rotation_trainloader = torch.utils.data.DataLoader(Rotation_trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2) # 64
    Rotation_testloader = torch.utils.data.DataLoader(Rotation_testset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2) # 32
    Make_batchloader = torch.utils.data.DataLoader(Rotation_testset, batch_size=1, shuffle=False, num_workers=2) # 32
    
    ## define model
    model = create_model(
                modelname = cfg['modelname'], 
                num_classes = data_stat['num_classes'], 
                img_size = data_stat['img_size'], 
                pretrained = False
            )
    
    ## define cuda and device    
    net = model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    else:
        net = net
        
    ## experiments setting
    criterion = nn.CrossEntropyLoss()
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[cfg['OPTIMIZER']['opt_name']](model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])
    
    # rotation train
    rotation_train(
        model = net,
        trainloader = Rotation_trainloader,
        testloader = Rotation_testloader,
        criterion = criterion,
        optimizer = optimizer,
        scheduler = scheduler, 
        save_path = cfg['save_path'],
        epoch = cfg['epoch']
    )
    
    ## make pt4al batch
    make_batch(
        model = net,
        testloader = Make_batchloader,
        criterion = criterion,  
        save_path = cfg['save_path']
    )