from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms
from spikingjelly.activation_based import neuron, functional, surrogate

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_updates = 1
thresh = 0.35
decay = 0.4
num_classes = 10
batch_size = 100
num_epochs = 30
tau_w = 40
lp_learning_rate = 5e-4
gp_learning_rate = 1e-3
time_window = 10
w_decay = 0.95
cfg_fc = [512, 10]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SNN_Model(nn.Module):
    def __init__(self):
        super(SNN_Model, self).__init__()
        
        self.fc1 = nn.Linear(28 * 28, cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        
        # Initialize neuronal parameters
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        
        # Initialize learnable parameters
        self.register_parameter('alpha1', nn.Parameter(1e-2 * torch.rand(1)))
        self.register_parameter('alpha2', nn.Parameter(1e-2 * torch.rand(1)))
        self.register_parameter('eta1', nn.Parameter(1e-2 * torch.rand(1, cfg_fc[0])))
        self.register_parameter('eta2', nn.Parameter(1e-2 * torch.rand(1, cfg_fc[1])))
        self.register_parameter('beta1', nn.Parameter(1e-2 * torch.rand(1, 28 * 28)))
        self.register_parameter('beta2', nn.Parameter(1e-2 * torch.rand(1, cfg_fc[0])))

    def reset_state(self):
        self.lif1.reset()
        self.lif2.reset()

    def compute_hebb_update(self, hebb, pre_spike, post_mem, beta, eta):
        pre_contrib = (pre_spike * beta).unsqueeze(2)
        post_contrib = ((post_mem / thresh) - eta).tanh().unsqueeze(1)
        hebb_delta = torch.bmm(pre_contrib, post_contrib).mean(dim=0).squeeze()
        new_hebb = w_decay * hebb + hebb_delta
        return torch.clamp(new_hebb, min=-4, max=4)

    def single_step_forward(self, x, hebb1, hebb2, step):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1).float()
        decay_factor = torch.tensor(np.exp(-step / tau_w), device=x.device)
        x_flat = x_flat * decay_factor
        
        # Layer 1
        h1_mem = self.fc1(x_flat) + self.alpha1 * torch.mm(x_flat, hebb1)
        h1_spike = self.lif1(h1_mem)
        
        # Layer 2
        h2_mem = self.fc2(h1_spike) + self.alpha2 * torch.mm(h1_spike, hebb2)
        h2_spike = self.lif2(h2_mem)
        
        # Compute new Hebbian weights
        new_hebb1 = self.compute_hebb_update(hebb1, x_flat, h1_mem, self.beta1, self.eta1)
        new_hebb2 = self.compute_hebb_update(hebb2, h1_spike, h2_mem, self.beta2, self.eta2)
        
        return h1_spike, h2_spike, new_hebb1, new_hebb2

    def forward(self, x, hebb1, hebb2):
        self.reset_state()
        batch_size = x.size(0)
        h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=x.device)
        h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=x.device)
        
        current_hebb1 = hebb1
        current_hebb2 = hebb2
        
        for step in range(time_window):
            h1_spike, h2_spike, new_hebb1, new_hebb2 = self.single_step_forward(
                x, current_hebb1, current_hebb2, step
            )
            
            h1_sumspike = h1_sumspike + h1_spike
            h2_sumspike = h2_sumspike + h2_spike
            
            current_hebb1 = new_hebb1
            current_hebb2 = new_hebb2

        return h2_sumspike, h1_sumspike, current_hebb1, current_hebb2
    
    def produce_hebb(self):
        hebb1 = torch.zeros(28 * 28, cfg_fc[0], device=device)
        hebb2 = torch.zeros(cfg_fc[0], cfg_fc[1], device=device)
        return hebb1, hebb2

def train_epoch(model, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    hebb1, hebb2 = model.produce_hebb()
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels_ = torch.zeros(batch_size, 10, device=device)
        labels_ = labels_.scatter_(1, labels.to(device).view(-1, 1), 1)
        
        optimizer.zero_grad()
        
        # Forward pass with detached Hebbian weights
        outputs, _, new_hebb1, new_hebb2 = model(images, hebb1.detach(), hebb2.detach())
        
        # Compute loss
        loss = criterion(outputs, labels_)
        loss += 0.001 * (torch.norm(model.eta1) + torch.norm(model.eta2))
        
        # Backward pass with retain_graph=True
        loss.backward(retain_graph=True)
        
        # Update weights
        optimizer.step()
        
        # Update Hebbian weights for next iteration
        hebb1 = new_hebb1.detach()
        hebb2 = new_hebb2.detach()
        
        running_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}], Step [{i+1}], Loss: {loss.item():.4f}')
    
    return running_loss

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    hebb1, hebb2 = model.produce_hebb()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs, _, _, _ = model(images, hebb1, hebb2)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def main():
    setup_seed(111)
    model = SNN_Model().to(device)
    
    # Data loading
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.FashionMNIST(
        root='../data', train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='../data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=gp_learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, epoch)
        accuracy = evaluate(model, test_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    main()