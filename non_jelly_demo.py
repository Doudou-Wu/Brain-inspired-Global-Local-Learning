from __future__ import print_function
import torch,time,os
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import torchvision
import torchvision.transforms as transforms
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np
import random
import  matplotlib.pyplot as plt

num_updates = 1 # meta-parameter update epochs, not used in this demo
thresh = 0.35 # threshold
lens = 0.5 # hyper-parameter in the approximate firing functions
decay = 0.4  # the decay constant of membrane potentials
num_classes = 10
batch_size = 100
num_epochs = 30
tau_w = 40 # synaptic filtering constant
lp_learning_rate = 5e-4  # learning rate of meta-local parameters
gp_learning_rate = 1e-3 # learning rate of gp-based parameters
time_window = 10 # time windows, we set T = 8 in our paper
w_decay = 0.95 # weight decay factor
cfg_fc = [512, 10] # Network structure

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ActFun(torch.autograd.Function):
    '''
    Approaximation function of spike firing rate function
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()


probs = 0.0 # dropout rate
act_fun = ActFun.apply

def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch>1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


class SNN_Model(nn.Module):

    def __init__(self ):
        super(SNN_Model, self).__init__()

        self.fc1 = nn.Linear(28*28, cfg_fc[0], )
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], )

        # alpha: the weight of hebb module
        self.alpha1 = torch.nn.Parameter((1e-2 * torch.rand(1)).cuda(), requires_grad=True)
        self.alpha2 = torch.nn.Parameter((1e-2 * torch.rand(1)).cuda(), requires_grad=True)

        # eta: the meta-local parameters  of sliding threshold
        self.eta1 = torch.nn.Parameter((1e-2* torch.rand(1,cfg_fc[0])).cuda(), requires_grad=True)
        self.eta2 = torch.nn.Parameter((1e-2 * torch.rand(1,cfg_fc[1])).cuda(), requires_grad=True)

        # gamma: the meta-local parameters to control the weight decay
        # not used in this demo
        self.gamma1 = torch.nn.Parameter((torch.rand(1)).cuda(), requires_grad=True)
        self.gamma2 = torch.nn.Parameter((torch.rand(1)).cuda(), requires_grad=True)

        # beta: the meta-local parameters to control the learning rate
        self.beta1 = torch.nn.Parameter((1e-2 * torch.rand(1, 784)).cuda(), requires_grad=True)
        self.beta2 = torch.nn.Parameter((1e-2 * torch.rand(1, cfg_fc[0])).cuda(), requires_grad=True)


    def mask_weight(self):
        self.fc1.weight.data = self.fc1.weight.data * self.mask1
        self.fc2.weight.data = self.fc2.weight.data * self.mask2

    def produce_hebb(self):
        hebb1 = torch.zeros(784, cfg_fc[0], device=device)
        hebb2 = torch.zeros(cfg_fc[0], cfg_fc[1], device=device)
        return hebb1, hebb2

    def parameter_split(self):
        '''
        Split the meta-local parameters and gp-based parameters for different update methods
        '''
        base_param = []
        for n, p in self.named_parameters():
            if n[:2] == 'fc' or n[:2] == 'fv':
                base_param.append(p)

        local_param = list(set(self.parameters()) - set(base_param))
        return base_param, local_param


    def forward(self, input,hebb1, hebb2, wins = time_window):

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        # hebb1 = torch.zeros(784, cfg_fc[0], device=device)

        for step in range(wins):

            decay_factor = np.exp(- step / tau_w)

            x = input
            # x = input > torch.rand(input.size(), device = device) # generate spike trains
            x = x.view(batch_size, -1).float()

            h1_mem, h1_spike, hebb1= mem_update_plastic(self.fc1,  self.alpha1, self.beta1, self.gamma1,self.eta1,
                                          x*decay_factor, h1_spike, h1_mem, hebb1)

            h1_sumspike = h1_sumspike + h1_spike

            h2_mem, h2_spike, hebb2 = mem_update_plastic(self.fc2,  self.alpha2,  self.beta2, self.gamma2,self.eta2,
                                          h1_spike*decay_factor, h2_spike, h2_mem, hebb2)

            h2_sumspike = h2_sumspike + h2_spike

        outs = h2_mem/thresh

        return outs.clamp(max = 1.1), h1_sumspike, h2_sumspike, hebb1.data, hebb2.data, self.eta1, self.eta2


def mem_update_plastic(fc, alpha, beta, gamma, eta, inputs,  spike, mem, hebb):
    '''
    Update the membrane potentials
    Note that : The only difference between the GP and HP model is whether to use hebb-based local variables
    :param fc: linear opetrations
    :param alpha: the weight of hebb module
    :param beta: the meta-local parameters to control the learning rate
    :param gamma: the meta-local parameters to control the weight decay, not used in this demo
    :param eta: the meta-local parameters  of sliding threshold
    :return: current membrane potentials, spikes, and local states
    '''
    state = fc(inputs) + alpha * inputs.mm(hebb)
    mem = (mem - spike * thresh) * decay + state
    now_spike = act_fun(mem - thresh)
    # Update local modules
    hebb = w_decay * hebb + torch.bmm((inputs * beta).unsqueeze(2), ((mem/thresh) - eta).tanh().unsqueeze(1)).mean(dim=0).squeeze()
    hebb = hebb.clamp(min = -4, max = 4)
    return mem, now_spike.float(), hebb



parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, default=r'../data')
parser.add_argument('--f', type=str, default='../Checkp-fashion/baseline')
parser.add_argument('--names', type=str, default='mlp_v1')

opt = parser.parse_args()

data_path = opt.p
save_path = opt.f
names = opt.names
lambdas = 0.

train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True,
                                                  transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True,
                                             transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0.  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
spi_record = list([])
loss_train_record = list([])
loss_test_record = list([])

criterion = nn.MSELoss()

total_best_acc = []
total_acc_record = []
total_hid_state = []

list_alpha1 = []
list_alpha2 = []
list_beta1 = []
list_beta2 = []
list_eta1 = []
list_eta2 = []

exp_num = 1
for exp in range(exp_num):
    setup_seed(111)
    snn = SNN_Model()
    snn.to(device)
    # optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)
    param_base, param_local = snn.parameter_split()
    optim_base = torch.optim.Adam(param_base, lr=1e-3)
    optim_local = torch.optim.Adam(param_local, lr=5e-4)

    acc_record = []
    hebb1, hebb2 = snn.produce_hebb()
    for epoch in range(num_epochs):
        running_loss = 0.
        snn.train()
        start_time = time.time()

        total = 0.
        correct = 0.

        for i, (images, labels) in enumerate(train_loader):
            snn.zero_grad()
            images = images.float().to(device)

            for i_update in range(num_updates):
                optim_base.zero_grad()
                outputs, spikes, _, hebb1, hebb2, eta1, eta2 = snn(input=images, hebb1=hebb1, hebb2=hebb2,
                                                                   wins=time_window)
                labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1) #label变成one-hot

                loss_reg = torch.norm(eta1, p=2) + torch.norm(eta2, p=2)
                loss = criterion(outputs.cpu(), labels_) + lambdas * loss_reg.cpu()
                loss.backward()
                optim_base.step()
                if i_update < (num_updates - 1): optim_local.zero_grad()

            optim_local.step()
            optim_local.zero_grad()
            running_loss += loss.item()

        print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f' % (
        epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, running_loss))
        print('Runing time:', time.time() - start_time)
        start_time = time.time()
        correct = 0.
        total = 0.
        running_loss = 0.

        # optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
        snn.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                # optimizer.zero_grad()
                outputs, sumspike, _, _, _, _, _ = snn(input=inputs, hebb1=hebb1, hebb2=hebb2, wins=time_window)

                labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
                loss = criterion(outputs.cpu(), labels_)

                _, predicted = outputs.cpu().max(1)
                total += float(targets.size(0))

                correct += float(predicted.eq(targets).sum().item())

            acc = 100. * float(correct) / float(total)

        print('Iters:', epoch, '\n')
        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))

        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)
        spi_record.append(sumspike.mean().detach())