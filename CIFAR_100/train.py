"""running models on MNIST"""
import json
import logging
import time

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms,datasets

from networks import block_selector
from utils import AverageMeter,ProgressMeter,CheckpointMeter,EarlyStopping,Param_tracker
from utils import accuracy,logset

"""program settings """
path_config = "./CIFAR_100/config.json"
path_log = './CIFAR_100/log.txt'
path_tensorboard = './CIFAR_100/tsboardlog/'

"""utils settings"""
logset(path_log)

"""load configs"""
file_config = open(path_config,'r')
configs = json.load(file_config)
file_config.close()

network = configs['networkname']
block = configs['blockname']
path_dataset = configs['dataset_path']
workers = configs['workers']
epochs = configs['epochs']
batch_size = configs['batch_size']
lr = configs['learning_rate']
weight_decay = configs['weight_decay']
path_save = configs['save_path']

logging.info("config loading complete")
tracker = Param_tracker(log_dir = path_tensorboard+block+'/',mode='layer4.3.binary')

"""dataset loading"""
normalize = transforms.Normalize(mean=(0.5,),std=(0.5,))
crop_scale = 0.08
#ighting_param = 0.1
train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees = 20),
    transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize])

val_transforms = transforms.Compose([
    transforms.RandomRotation(degrees = 20),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize])

train_dataset = torchvision.datasets.CIFAR100(
    root=path_dataset, train=True, 
    download=True, 
    transform=train_transforms)

val_dataset = torchvision.datasets.CIFAR100(
    root=path_dataset, train=False, 
    download=True, 
    transform=val_transforms)

train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)    
val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, shuffle=False,
        num_workers=workers,pin_memory=True)

logging.info('dataset loading complete')
"""main train progress"""
def main():
    start_t = time.time()

#initialize model and loss function
    model = model_selector(network,block)
    model = nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    logging.info('model and criterion initialized')
#separate weight and other parameters
    all_parameters = model.parameters()
    #print (list(all_parameters))
    weight_parameters = []
    
    for pname, p in model.named_parameters():
        # print(pname)
        # print(p.shape)
        if p.ndimension() == 4 or 'conv' in pname:
            weight_parameters.append(p)
    
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))
   
    tracker.param_read(model.named_parameters())
    #tracker.params_check()

#initialize optmizer
    optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' :weight_decay}],
            lr=lr,)
    
#initialize scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode='min',factor=0.2,patience=3,verbose=False,
        threshold=0.0001,threshold_mode='rel',
        cooldown=0,min_lr=0,eps=1e-08)

    earlystop = EarlyStopping(
        patience=9,verbose=False,delta = 0)

    logging.info('optiimizer and scheduler initialized')
#load checkpoints
    ckptmeter = CheckpointMeter(path_save,block,save_best=True)
    checkpoint,start_epoch,best_top1_acc = ckptmeter.load()
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'],strict=False)
        start_epoch = start_epoch + 1#the epoch 'start_epoch' has been trained already
        logging.info("weight loaded, epoch = {}" .format(checkpoint['epoch']))

    epoch = start_epoch
    while epoch < epochs:
    #training process
        # extract(epoch, train_loader, model, criterion, optimizer, scheduler)
        # exit()
        train_obj, train_top1_acc = train(epoch, train_loader, model, criterion, optimizer, scheduler)
        valid_loss, valid_top1_acc = validate(epoch, val_loader, model, criterion,0)
    #update scheduler
        scheduler.step(valid_loss)#active this when using ReduceLROnPlateau scheduler
        
    #save checkpoints
        ckptmeter.save(epoch,model,valid_top1_acc,optimizer)
    #output info
        training_time = (time.time() - start_t) / 3600
        print('total training time = {} hours'.format(training_time))
        logging.info('epoch {} accomplished, loss = {}, acc = {}'
            .format(epoch,valid_loss,valid_top1_acc))

        epoch += 1

    #early stop
        if earlystop.early_stop:
            print("early stopped,","epoch = ",epoch)
            break

"""train and validate function"""
def train(epoch, train_loader, model, criterion, optimizer, scheduler):
#prepare to record
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    
#adjust parameters
    model.train()
    #scheduler.step() #ban it when using ReduceLROnPlateau scheduler
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

#steps
    j = 0
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()
       
    # compute outputy
        pred = model(images) 
        #print(pred.size())
        loss = criterion(pred,target)

    # measure accuracy and record loss
        prec1, prec5 = accuracy(pred, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)

    # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    #output
        progress.display(i)
        #tracker.params_check()
        j += 1
        if j == 50:
            tracker.update(option=['abs_mean','histogram_all'],grad=[True,False])
            logging.info("batch{}, acc:{}, loss:{}"
            .format(i,prec1,loss))
            j = 0
    
    return losses.avg, top1.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        ##steps
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, _ = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)

        print(' * acc@1 {top1.avg:.3f} '.format(top1=top1))

    return losses.avg, top1.avg

def extract(epoch, train_loader, model, criterion, optimizer, scheduler):
    """run an epoch to extract the weight info and so on"""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    
#adjust parameters
    model.train()
    #scheduler.step() #ban it when using ReduceLROnPlateau scheduler
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    record_count = 0
#steps
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()
       
    # compute outputy
        pred = model(images) 
        #print(pred.size())
        loss = criterion(pred,target)

    # measure accuracy and record loss
        prec1, prec5 = accuracy(pred, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)

    # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    #output
        progress.display(i)
        if record_count >= 100:
            tracker.update(
                option=['abs_mean','histogram_all','abs_mean','histogram_all'],
                grad=[True,True,False,False]) 
            record_count
        record_count += 1
    return losses.avg, top1.avg
"""other functions"""
def model_selector(name,arg):
    if name == "resnet":
        return block_selector(arg)

if __name__ == '__main__':
    main()