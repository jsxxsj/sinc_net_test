import os
import numpy as np
import logging
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import shutil

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class CheckpointMeter(object):
    """a class for load,save and update checkpoints"""
    def __init__(self,save_path,ckpt_name = 'ckpt.pth.tar',save_best = True) -> None:
        
        self.path = save_path

        #adjust pathes
        if not str.endswith(ckpt_name,'.pth.tar'):
            ckpt_name += '.pth.tar'
        self.ckpt_full = os.path.join(save_path,ckpt_name)
        best_name = str.replace(ckpt_name,'.pth.tar','_best.pth.tar')
        self.best_full = os.path.join(save_path,best_name)

        #create dir
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        #acc for storing best ckpt
        self.acc = 0 

    def load(self):
        """load weights before training
        output:checkpoint,start_epoch,best_top1_acc
        checkpoint can be load directly, and will be None if no file exist"""
        if not os.path.exists(self.ckpt_full):
            checkpoint = None
            start_epoch = 0
            best_top1_acc = 0
        else:
            checkpoint = torch.load(self.ckpt_full)
            start_epoch = checkpoint['epoch']
            best_top1_acc = checkpoint['best_top1_acc']
        return checkpoint,start_epoch,best_top1_acc
    
    def save(self,epoch,model,acc,optimizer):
        """
        save weights after training
        """
        #pack the states required
        state = {
            'epoch':epoch,
            'state_dict':model.state_dict(),
            'best_top1_acc':acc,
            'potimizer':optimizer.state_dict()}
        
        #save the files
        torch.save(state, self.ckpt_full)
        if acc>self.acc:
            shutil.copyfile(self.ckpt_full, self.best_full)
            self.acc = acc

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='./earlystop/checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def logset(logpath):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(logpath))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

class Param_tracker():
    """using tensorboard to track the features of selected params in a model"""
    def __init__(self,log_dir= None,mode = 'conv') -> None:
        # self.means = []
        # self.stds = []
        self.mode = mode
        self.writer = SummaryWriter(log_dir= log_dir)
        self.step = 0
    
    def param_read(self,param):
        """please use tModule.named_weights()"""
        self.params = []
        self.names = []

        for pname,p in param:
            if self.mode == 'all':
                self.params.append(p)
                self.names.append(pname)

            elif self.mode == 'weight':
                if p.ndimension() == 4 or 'conv' in pname:
                    self.params.append(p)
                    self.names.append(pname)
            
            else:
                if self.mode in pname:
                    self.params.append(p)
                    self.names.append(pname)
    
    def params_check(self):
        for i in range (0,len(self.names)):
            print(self.names[i])
            print(self.params[i].shape)

    def update(self,option = ['mean','std'],grad = [False,False]):
        #compute grad
        grad_params = []
        for p in self.params:
            grad_params.append(p.grad)
            cated_params = torch.cat(grad_params)

        for i, a_option in enumerate( option):
            if grad[i]:
                tag0 = self.mode + '_grad'
                cated_params = torch.cat(grad_params)
            else:
                tag0 = self.mode
                cated_params = torch.cat(self.params) 

            if a_option == 'mean':
                self.writer.add_scalar(
                    tag=tag0+'_mean',
                    scalar_value= torch.mean(cated_params),
                    global_step=self.step)
            
            if a_option == 'std':
                self.writer.add_scalar(
                    tag=tag0+'_std',
                    scalar_value= torch.std(cated_params),
                    global_step=self.step)
            
            if a_option == 'abs_mean':
                self.writer.add_scalar(
                    tag=tag0+'_abs_mean',
                    scalar_value= torch.mean(torch.abs(cated_params)),
                    global_step=self.step)
            
            if a_option == 'histogram':
                self.writer.add_histogram(
                    tag = tag0 +'_histogram',
                    values=cated_params,
                    global_step= self.step)
            
            if a_option == 'histogram_all':
                for j,p in enumerate(self.params):
                    tag1 = self.names[j]
                    if grad[i]: 
                        tag1 += '_grad'
                        values = p.grad
                    else:
                        values = p

                    self.writer.add_histogram(
                        tag = tag1 +'_histogram',
                        values= values,
                        global_step = self.step
                    )
        self.step += 1

class test():
    def __init__(self) -> None:
        pass
    def read(self,a):
        self.a = torch.cat([a,torch.zeros(a.size()).cuda()])
    def print(self):
        print(torch.std(self.a))



        
            


    

        