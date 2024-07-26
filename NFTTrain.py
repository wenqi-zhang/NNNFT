# 128 Channel 16-QAM Freq to NFT with random carrier wave including (different flattop, sinc, q-modulation and b-modulation)

import torch
import torch.nn as nn
import torch.optim as optim
from NFTDataset import NFTDataset
from torch.utils.data import DataLoader, ConcatDataset
import os
import tqdm as tqdm
import matplotlib.pyplot as plt
from utils import matplot, plotresult, plotresult_
from NFTModel import ConvLSTMAutoEncoder
import numpy as np

mfname='model'+os.path.splitext(os.path.basename(__file__))[0]+'.pth'

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

plt.figure(1)
plt.clf()
training_loss_plotter, = plt.semilogy([],[],'.-',markersize=1,linewidth=0.5)
test_loss_plotter, = plt.semilogy([],[],'x--')
plt.grid(True,'both')

#########################################################################################

lr = 3e-4
batch_size = 200

training_data1 = NFTDataset('DataSet128p1.mat',0,0.01,normalization='Q',transform=torch.tensor)
training_data2 = NFTDataset('DataSet128p2.mat',0,0.01,normalization='Q',transform=torch.tensor)
test_data1 = NFTDataset('DataSet128p1.mat',1,0.01,normalization='Q',transform=torch.tensor)
test_data2 = NFTDataset('DataSet128p2.mat',1,0.01,normalization='Q',transform=torch.tensor)
training_data = ConcatDataset([training_data1, training_data2])
test_data = ConcatDataset([test_data1, test_data2])
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

nt = training_data[0][0].shape[1]
nc = training_data[0][0].shape[0]

model = ConvLSTMAutoEncoder(nc, nc, nt, kernel_size=3, padding=1, stride=2, features=[64, 128, 256], lstm_layers=1, forward=False).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scaler = torch.cuda.amp.GradScaler()

#########################################################################################

def loss_fn(input, target):
    return torch.sqrt(nn.functional.mse_loss(input, target))

def train(model, optimizer, dataloader):
    global iterations
    model.train()
    pbar = tqdm.tqdm(dataloader, disable=False)
    for x, y, _ in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        if device=="cuda":
            with torch.cuda.amp.autocast():
                y_hat = model(x)
                loss = loss_fn(y_hat, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

        pbar.set_postfix(loss="{:.3f}".format(loss.item()))
        iterations += len(x)/dataloader.dataset.__len__()
        matplot(training_loss_plotter,iterations,loss.item())
        
        if (iterations * dataloader.dataset.__len__() // batch_size) % 10 == 0:
            model.eval()
            with torch.no_grad():
                plotresult(model, test_data)
            model.train()

def valid(model, dataloader):
    global iterations
    model.eval()
    with torch.no_grad():
        for x, y, _ in dataloader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
        matplot(test_loss_plotter,iterations,loss.item())

iterations = 0
epochs = 200
epoch = 0

try:
    print('Load previously saved model: ' + mfname + ' ...')
    savedata = torch.load(mfname, map_location=device)
    model.load_state_dict(savedata['model'], strict=False)
    optimizer.load_state_dict(savedata['optimizer'])
    scaler.load_state_dict(savedata['scaler'])
    epoch = savedata['epoch']
    training_loss_plotter.set_xdata(savedata['trainingloss'][:,0])
    training_loss_plotter.set_ydata(savedata['trainingloss'][:,1])
    training_loss_plotter.axes.relim()
    training_loss_plotter.axes.autoscale()
    test_loss_plotter.set_xdata(savedata['testloss'][:,0])
    test_loss_plotter.set_ydata(savedata['testloss'][:,1])
    test_loss_plotter.axes.relim()
    test_loss_plotter.axes.autoscale()
    plt.draw()
    plt.pause(0.01)
    iterations = savedata['iterations']
except:
    print("Failed to load saved model, start anew!")

optimizer.param_groups[0]['lr'] = lr

for epoch in range(epoch+1,epochs+1):
    print(f"Epoch {epoch}\n-------------------------------")
    train(model, optimizer, train_dataloader)
    valid(model, test_dataloader)
    savedict = {
        'model':model.state_dict(), 
        'optimizer':optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch':epoch,
        'trainingloss':training_loss_plotter.get_xydata(),
        'testloss':test_loss_plotter.get_xydata(),
        'iterations':iterations
    }
    torch.save(savedict, mfname)
    if epoch % 10 == 0:
        torch.save(savedict, mfname + '.' + str(epoch))
    print("Saved PyTorch Model State to " + mfname)
    
print("Done!")

model.eval()
with torch.no_grad():
    plotresult(model, test_data)
   