import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.backends.mps


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class ChessValueDataset(Dataset):
    def __init__(self):
        data =np.load("processed/dataset_100K.npz")
        #print(data['arr_0'])
        #print(data['arr_1'])
        self.X = data['arr_0']
        self.Y = data['arr_1']
        print('Loaded !  X shape: ', self.X.shape, 'Y shape: ',self.Y.shape) 

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'Y': self.Y[idx]}



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.a1 = nn.Conv2d(5, 16, kernel_size=3, padding=2)
        self.a2 = nn.Conv2d(16, 32, kernel_size=3, padding=2)
        self.a3 = nn.Conv2d(32, 64, kernel_size=3, padding=2)

        self.b1 = nn.Conv2d(64, 32, kernel_size=3, padding=2)
        self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=2)
        self.b3 = nn.Conv2d(32, 64, kernel_size=3, padding=2)

        self.c1 = nn.Conv2d(64, 64, kernel_size=3, padding=2)
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, padding=2)
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, padding=2)

        self.d1 = nn.Conv2d(128, 128, kernel_size=3, padding=2)
        self.d2 = nn.Conv2d(128, 128, kernel_size=3, padding=2)
        self.d3 = nn.Conv2d(128, 128, kernel_size=3, padding=2)

        # Dummy forward pass to calculate the size dynamically
        self._to_linear = None
        self._dummy_x = torch.zeros(1, 5, 8, 8)
        self._calculate_to_linear()

        self.lost = nn.Linear(4608, 1)

    def _calculate_to_linear(self):
        with torch.no_grad():
            self._dummy_x = self._conv_pass(self._dummy_x)
            self._to_linear = self._dummy_x.view(-1).shape[0]

    def _conv_pass(self, x):
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))
        x = F.max_pool2d(x, kernel_size=2)

        return x

    def forward(self, x):
        x = x.view(-1, 5, 8, 8)
        x = self._conv_pass(x)
        x = x.view(x.size(0), -1)
        x = self.lost(x)
        return x.squeeze()

    


chess_dataset = ChessValueDataset() 
model = Net().to(device)
#model.float()

#create a Adam optimizer
optimizer = optim.Adam(model.parameters(),lr=0.01)

#Load the dataset to torch DataLoader
train_loader = DataLoader(chess_dataset,batch_size=32,shuffle=True)

# Instantiate the loss function
loss_function = nn.MSELoss()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        X = data['X'].float().to(device)
        target = data['Y'].float().to(device)

        optimizer.zero_grad()
        output = model(X)
        output = output.squeeze(-1)  # Adjust this line to match target shape


        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(X)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    print(f'Epoch: {epoch}')


for epoch in range(1,10):
    train(model,device,train_loader,optimizer,epoch)
    torch.save(model.state_dict(),"models/chess_model_1.pt")
 
    
