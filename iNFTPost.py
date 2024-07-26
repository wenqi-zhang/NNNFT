import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
import os
import psutil
import gc
import matplotlib.pyplot as plt
from utils import *
from NFTModel import ConvLSTMAutoEncoder
from NFTDataset import NFTDataset
import numpy as np

def mem_usage():
    process = psutil.Process(os.getpid())
    mu = process.memory_info().rss / 1024**2
    print('Memory usage: {:.2f} MB'.format(mu))

gc.collect()
torch.cuda.empty_cache()

mem_usage()

mfname='modelinfttrain.pth'

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

batch_size = 200
dataset = NFTDataset('DataSet128p3.mat', 0, 0, normalization='U', transform=torch.tensor)

nt = dataset[0][0].shape[1]
nc = dataset[0][0].shape[0]

model = ConvLSTMAutoEncoder(nc, nc, nt, kernel_size=3, features=[64, 128, 256], lstm_layers=1, forward=True).to(device)

try:
    print('Load previously saved model: ' + mfname + ' ...')
    savedata = torch.load(mfname, map_location=device)
    model.load_state_dict(savedata['model'], strict=False)
except:
    print("Failed to load saved model!")

def loss_fn(input, target):
    return torch.sqrt(torch.mean(nn.functional.mse_loss(input, target, reduction='none'), [1, 2]))

def plotSpectra(lam, U, V, Q):
    plt.figure()
    plt.subplot(3,2,1)
    plt.plot(lam, U[0].cpu().numpy(), '.-', markersize=0.5, linewidth=0.1)
    plt.subplot(3,2,2)
    plt.plot(lam, U[1].cpu().numpy(), '.-', markersize=0.5, linewidth=0.1)
    plt.subplot(3,2,3)
    plt.plot(lam, V[0].cpu().numpy(), '.-', markersize=0.5, linewidth=0.1)
    plt.subplot(3,2,4)
    plt.plot(lam, V[1].cpu().numpy(), '.-', markersize=0.5, linewidth=0.1)
    plt.subplot(3,2,5)
    plt.plot(lam, Q[0].cpu().numpy(), '.-', markersize=0.5, linewidth=0.1)
    plt.subplot(3,2,6)
    plt.plot(lam, Q[1].cpu().numpy(), '.-', markersize=0.5, linewidth=0.1)

nrange = 10

model.eval()
with torch.no_grad():
    
## Sample spectra with different num of channels and different power levels
    Uch = []
    Qch = []
    Ech = []

    X = []
    Y = []
    Y_hat = []
    
    for ii in range(3):
        uii = dataset.u[:, (dataset.ch==ii) & (dataset.aa==2)]
        qii = dataset.q[:, (dataset.ch==ii) & (dataset.aa==2)]
        Uii, Qii, (Eii, _, _, _) = SeperatePulseByEnergyLogRanges_(uii, qii, nrange)
        # Uch.append(Uii)
        # Qch.append(Qii)
        # Ech.append(Eii)

        Xii = []
        Yii = []
        Yii_hat = []
        for jj in (0, 9):
            # meanE = np.mean(Eii[jj])
            # idx = np.abs(Eii[jj] - meanE).argmin()
            idx = np.random.randint(len(Eii[jj]))
            y, y_norm = prepareU(Uii[jj][:,idx], normalization=True)
            x, x_norm = prepareQ(Qii[jj][:,idx], normalization=False)
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            Xii.append(x[0].cpu().numpy())
            Yii.append(y[0].cpu().numpy())
            Yii_hat.append((y_hat[0]).cpu().numpy())
            plotSpectra(dataset.lam, x[0] * x_norm, y_hat[0] * y_norm, y[0] * y_norm)

        X.append(Xii)
        Y.append(Yii)
        Y_hat.append(Yii_hat)
    
    loss, losses = valid(model, dataset, loss_fn, batch_size=200, forward=True)
    # loss = LossVsEnergyLog(model, dataset, loss_fn, nrange)
    E = np.log10(GetEnergy(dataset.u))
    Emin, Emax = np.min(E), np.max(E)
    dE = (Emax - Emin) / nrange
    
    loss_E = []
    meanE_E = []
    E1 = Emin
    for ii in range(nrange):
        E2 = E1 + dE
        eii = np.mean(10.0**E[(E >= E1) & (E < E2)])
        meanE_E.append(eii)
        lii = np.mean(losses[(E >= E1) & (E < E2)])
        loss_E.append(lii)
        E1 = E2

    loss_ch = []
    meanE_ch = []
    for ii in range(3):
        lii = np.mean(losses[dataset.ch==ii])
        eii = np.mean(10.0**E[dataset.ch==ii])
        loss_ch.append(lii)
        meanE_ch.append(eii)

    loss_aa = []
    meanE_aa = []
    for ii in range(3):
        lii = np.mean(losses[dataset.aa==ii])
        eii = np.mean(10.0**E[dataset.aa==ii])
        loss_aa.append(lii)
        meanE_aa.append(eii)

savedict = {
    'lam' : dataset.lam,
    'X' : X,
    'Y' : Y,
    'Y_hat' : Y_hat,
    'losses' : losses,
    'E' : E,
    'loss_E' : loss_E,
    'meanE_E' : meanE_E,
    'loss_ch' : loss_ch,
    'meanE_ch' : meanE_ch,
    'loss_aa' : loss_aa,
    'meanE_aa' : meanE_aa
}

torch.save(savedict, 'ipostprocess.pth')

#plt.plot(loss)
