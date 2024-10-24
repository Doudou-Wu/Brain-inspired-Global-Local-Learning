import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate
from torchvision import datasets, transforms
import numpy as np
import random
import time

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
num_epochs = 30
time_window = 10
thresh = 0.35  
decay = 0.4    
tau_w = 40     
w_decay = 0.95 
cfg_fc = [512, 10]
lp_learning_rate = 5e-4
gp_learning_rate = 1e-3

class HebbianLIFNeuron(neuron.BaseNode):
    def __init__(self, size, alpha, beta, gamma, eta, surrogate_function=surrogate.ATan()):
        super().__init__(surrogate_function=surrogate_function)
        self.size = size
        self.alpha = alpha
        self.beta = beta  
        self.gamma = gamma
        self.eta = eta
        self.register_buffer('hebb', torch.zeros(size, dtype=torch.float32))
        self.reset()
        
    def reset(self):
        self.v = 0.0
        
    def neuronal_charge(self, x: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x)
            
        hebb_term = self.alpha * self.hebb.unsqueeze(0)
        # 使用新变量存储计算结果，避免inplace操作
        new_v = self.v * decay + x + hebb_term
        self.v = new_v

    def neuronal_fire(self):
        return self.surrogate_function(self.v - thresh)

    def update_hebb(self, x):
        with torch.no_grad():  # Hebbian更新不需要计算梯度
            v_normalized = (self.v/thresh) - self.eta
            delta_hebb = torch.mean(v_normalized, dim=0)
            new_hebb = w_decay * self.hebb + self.beta * delta_hebb
            self.hebb = torch.clamp(new_hebb, -4, 4)

    def forward(self, x):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.update_hebb(x)
        return spike

class HebbianSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = layer.Linear(28*28, cfg_fc[0])
        self.fc2 = layer.Linear(cfg_fc[0], cfg_fc[1])
        
        self.alpha1 = nn.Parameter(torch.rand(1) * 1e-2)
        self.alpha2 = nn.Parameter(torch.rand(1) * 1e-2)
        self.beta1 = nn.Parameter(torch.rand(1) * 1e-2)
        self.beta2 = nn.Parameter(torch.rand(1) * 1e-2)
        self.gamma1 = nn.Parameter(torch.rand(1))
        self.gamma2 = nn.Parameter(torch.rand(1))
        self.eta1 = nn.Parameter(torch.rand(cfg_fc[0]) * 1e-2)
        self.eta2 = nn.Parameter(torch.rand(cfg_fc[1]) * 1e-2)
        
        self.lif1 = HebbianLIFNeuron(cfg_fc[0], self.alpha1, self.beta1, 
                                    self.gamma1, self.eta1)
        self.lif2 = HebbianLIFNeuron(cfg_fc[1], self.alpha2, self.beta2, 
                                    self.gamma2, self.eta2)
    
    def reset(self):
        self.lif1.reset()
        self.lif2.reset()
        
    def forward(self, x, T=time_window):
        self.reset()
        batch_size = x.shape[0]
        
        # 初始化累积张量
        h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=x.device)
        h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=x.device)
        
        for t in range(T):
            decay_factor = torch.exp(-torch.tensor(t / tau_w, device=x.device))
            
            x_t = x.view(batch_size, -1) * decay_factor
            h1 = self.fc1(x_t)
            h1_spike = self.lif1(h1)
            
            h1_sumspike = h1_sumspike + h1_spike
            
            h2 = self.fc2(h1_spike * decay_factor)
            h2_spike = self.lif2(h2)

            h2_sumspike = h2_sumspike + h2_spike
            
            outputs = h2_spike
        
        return torch.clamp(outputs, max=1.1), h1_sumspike, h2_sumspike

def train(model, train_loader, test_loader):
    base_params = [p for name, p in model.named_parameters() if name.startswith('fc')]
    local_params = [p for name, p in model.named_parameters() if not name.startswith('fc')]
    
    optim_base = torch.optim.Adam(base_params, lr=gp_learning_rate)
    optim_local = torch.optim.Adam(local_params, lr=lp_learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.float().to(device)
            labels_onehot = torch.zeros(images.shape[0], 10, device=device)
            labels_onehot = labels_onehot.scatter(1, labels.to(device).view(-1, 1), 1)
            
            outputs, _, _ = model(images)
            loss = criterion(outputs, labels_onehot)
            
            optim_base.zero_grad()
            optim_local.zero_grad()
            loss.backward()
            optim_base.step()
            optim_local.step()
            
            running_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.float().to(device)
                labels = labels.to(device)
                outputs, _, _ = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.5f}, '
              f'Test Accuracy: {acc:.2f}%')
        print(f'Running time: {time.time() - start_time:.2f}s')

def main():
    setup_seed(111)
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(root='../data', train=True,
                                        download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='../data', train=False,
                                       download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False)
    
    model = HebbianSNN().to(device)
    train(model, train_loader, test_loader)

if __name__ == "__main__":
    main()