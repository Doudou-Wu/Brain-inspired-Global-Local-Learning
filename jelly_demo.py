from __future__ import print_function
import torch, time, os
import torch.nn as nn
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms
from spikingjelly.activation_based import neuron, functional, surrogate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_updates = 1  # meta-parameter update epochs
thresh = 0.35  # spike threshold
decay = 0.4  # membrane potential decay constant
num_classes = 10
batch_size = 100
num_epochs = 30
tau_w = 40  # synaptic filtering constant
lp_learning_rate = 5e-4  # learning rate for local parameters
gp_learning_rate = 1e-3  # learning rate for global parameters
time_window = 10  # time windows for simulation
w_decay = 0.95  # weight decay factor for Hebbian learning
cfg_fc = [512, 10]  # Fully connected layers structure

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

torch.autograd.set_detect_anomaly(True)

class SNN_Model(nn.Module):
    def __init__(self):
        super(SNN_Model, self).__init__()

        self.fc1 = nn.Linear(28 * 28, cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

        # Use SpikingJelly LIFNode
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan())

        # Hebbian module parameters (random initialization)
        self.alpha1 = torch.nn.Parameter((1e-2 * torch.rand(1)).cuda(), requires_grad=True)
        self.alpha2 = torch.nn.Parameter((1e-2 * torch.rand(1)).cuda(), requires_grad=True)

        # Meta-local parameters for sliding threshold
        self.eta1 = torch.nn.Parameter((1e-2 * torch.rand(1, cfg_fc[0])).cuda(), requires_grad=True)
        self.eta2 = torch.nn.Parameter((1e-2 * torch.rand(1, cfg_fc[1])).cuda(), requires_grad=True)

        # Meta-local parameters to control learning rate
        self.beta1 = torch.nn.Parameter((1e-2 * torch.rand(1, 28 * 28)).cuda(), requires_grad=True)
        self.beta2 = torch.nn.Parameter((1e-2 * torch.rand(1, cfg_fc[0])).cuda(), requires_grad=True)

    def forward(self, x, hebb1, hebb2, wins=time_window):
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        for step in range(wins):
            x_flat = x.view(batch_size, -1).float() #展平成二维向量
            decay_factor = np.exp(- step / tau_w)
            x_flat = x_flat * decay_factor
            
            # Layer 1: Linear -> LIF with Hebbian update
            h1_mem = self.fc1(x_flat) + self.alpha1 * torch.mm(x_flat, hebb1)
            hebb1 = self.update_hebb(hebb1, x_flat, h1_mem, self.beta1, self.eta1)
            # hebb1 = w_decay * hebb1 + torch.bmm((x_flat * self.beta1).unsqueeze(2), ((h1_mem / thresh) - self.eta1).tanh().unsqueeze(1)).mean(dim=0).squeeze()
            h1_spike = self.lif1(h1_mem)
            h1_sumspike =  h1_sumspike + h1_spike

            # Layer 2: Linear -> LIF with Hebbian update
            h2_mem = self.fc2(h1_spike) + self.alpha2 * torch.mm(h1_spike, hebb2)
            hebb2 = self.update_hebb(hebb2, h1_spike, h2_mem, self.beta2, self.eta2)
            h2_spike = self.lif2(h2_mem)
            h2_sumspike = h2_sumspike + h2_spike

        return h2_spike, h1_sumspike, h2_sumspike, hebb1, hebb2, self.eta1, self.eta2

    def update_hebb(self, hebb, pre_spike, post_mem, beta, eta):
        hebb = w_decay * hebb + torch.bmm((pre_spike * beta).unsqueeze(2), ((post_mem / thresh) - eta).tanh().unsqueeze(1)).mean(dim=0).squeeze()
        hebb = hebb.clamp(min=-4, max=4)
        return hebb

    def produce_hebb(self):
        hebb1 = torch.zeros(28 * 28, cfg_fc[0], device=device)
        hebb2 = torch.zeros(cfg_fc[0], cfg_fc[1], device=device)
        return hebb1, hebb2

# Data loading
train_dataset = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Training loop
setup_seed(111)
snn = SNN_Model()
snn.to(device)
param_base, param_local = snn.parameters(), []  # Use standard PyTorch parameter handling for both global and local
optim_base = torch.optim.Adam(param_base, lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    snn.train()
    hebb1, hebb2 = snn.produce_hebb()
    running_loss = 0.
    for i, (images, labels) in enumerate(train_loader):
        images = images.float().to(device)
        optim_base.zero_grad()

        outputs, _, _, hebb1, hebb2, eta1, eta2 = snn(x=images, hebb1=hebb1, hebb2=hebb2, wins=time_window)
        labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)
        
        loss_reg = torch.norm(eta1, p=2) + torch.norm(eta2, p=2)
        loss = criterion(outputs.cpu(), labels_) + loss_reg.cpu()
        loss.backward()
        optim_base.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss}")

    # Evaluate the model
    snn.eval()
    with torch.no_grad():
        total = correct = 0
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs, _, _, _, _, _, _ = snn(x=inputs, hebb1=hebb1, hebb2=hebb2, wins=time_window)
            _, predicted = outputs.cpu().max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        print(f"Test Accuracy: {acc:.2f}%")
