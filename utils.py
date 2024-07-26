import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

def matplot(plotter, x, y):
    plt.figure(1)
    plotter.set_xdata(np.append(plotter.get_xdata(), x))
    plotter.set_ydata(np.append(plotter.get_ydata(), y))
    plotter.axes.relim()
    plotter.axes.autoscale()
    plt.draw()
    plt.pause(0.01)

def plotresult(model, test_data):
    device = next(model.parameters()).device
    x, y, _ = test_data.__getitem__(int(np.random.rand()*test_data.__len__()))
    x = x.to(device)
    x = x.reshape(1,x.shape[0],x.shape[1])
    y = y.reshape(1,y.shape[0],y.shape[1])
    with torch.no_grad():
        y_ = model(x).cpu()
    x = x.cpu()
    plt.figure(2)
    plt.clf()
    plt.subplot(3,2,1)
    plt.plot(x[0][0].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(3,2,2)
    plt.plot(x[0][1].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(3,2,3)
    plt.plot(y_[0][0].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(3,2,4)
    plt.plot(y_[0][1].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(3,2,5)
    plt.plot(y[0][0].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(3,2,6)
    plt.plot(y[0][1].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.draw()
    plt.pause(0.01)

def plotresulti(model, test_data):
    device = next(model.parameters()).device
    y, x, _ = test_data.__getitem__(int(np.random.rand()*test_data.__len__()))
    x = x.to(device)
    x = x.reshape(1,x.shape[0],x.shape[1])
    y = y.reshape(1,y.shape[0],y.shape[1])
    with torch.no_grad():
        y_ = model(x).cpu()
    x = x.cpu()
    plt.figure(2)
    plt.clf()
    plt.subplot(3,2,1)
    plt.plot(x[0][0].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(3,2,2)
    plt.plot(x[0][1].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(3,2,3)
    plt.plot(y_[0][0].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(3,2,4)
    plt.plot(y_[0][1].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(3,2,5)
    plt.plot(y[0][0].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(3,2,6)
    plt.plot(y[0][1].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.draw()
    plt.pause(0.01)

def plotresulti2(model1, model2, test_data):
    device = next(model1.parameters()).device
    y, x, _ = test_data.__getitem__(int(np.random.rand()*test_data.__len__()))
    x = x.to(device)
    x = x.reshape(1,x.shape[0],x.shape[1])
    y = y.reshape(1,y.shape[0],y.shape[1])
    with torch.no_grad():
        x = model1(x)
        y_ = model2(x).cpu()
    x = x.cpu()
    plt.figure(2)
    plt.clf()
    plt.subplot(3,2,1)
    plt.plot(x[0][0].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(3,2,2)
    plt.plot(x[0][1].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(3,2,3)
    plt.plot(y_[0][0].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(3,2,4)
    plt.plot(y_[0][1].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(3,2,5)
    plt.plot(y[0][0].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(3,2,6)
    plt.plot(y[0][1].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.draw()
    plt.pause(0.01)

def plotresult_(model, x0):
    device = next(model.parameters()).device
    phaseComp = np.exp(1j*np.pi*np.arange(len(x0))) * 2e5
    x = -np.conj(np.complex64(np.fft.fftshift(np.fft.ifft(x0)) * phaseComp))
    xreal = np.real(x)
    ximag = np.imag(x)
    x = torch.tensor(np.c_[xreal,ximag].T).to(device)
    x = x.reshape(1,x.shape[0],x.shape[1])
    with torch.no_grad():
        y = model(x).cpu()
    x = x.cpu()
    plt.figure(3)
    plt.clf()
    plt.subplot(2,2,1)
    plt.plot(x[0][0].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(2,2,2)
    plt.plot(x[0][1].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(2,2,3)
    plt.plot(y[0][0].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(2,2,4)
    plt.plot(y[0][1].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.draw()
    plt.pause(0.01)
    return x[0][0].numpy()+1j*x[0][1].numpy(), y[0][0].numpy()+1j*y[0][1].numpy()

def plotresulti_(model, x0):
    device = next(model.parameters()).device
    x = np.complex64(x0);
    xreal = np.real(x)
    ximag = np.imag(x)
    x = torch.tensor(np.c_[xreal,ximag].T).to(device)
    x = x.reshape(1,x.shape[0],x.shape[1])
    with torch.no_grad():
        y = model(x).cpu()
    x = x.cpu()
    plt.figure(3)
    plt.clf()
    plt.subplot(2,2,1)
    plt.plot(x[0][0].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(2,2,2)
    plt.plot(x[0][1].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(2,2,3)
    plt.plot(y[0][0].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.subplot(2,2,4)
    plt.plot(y[0][1].numpy(),'.-',markersize=0.5,linewidth=0.1)
    plt.draw()
    plt.pause(0.01)
    return y[0][0].numpy()+1j*y[0][1].numpy()

def GetEnergy(dataU):
    nt = 2**11
    dt = 1.0/96e9
    T = dt*nt
    t = np.arange(nt) * dt - T / 2
    P0 = 2.0/1.1e-3
    E = np.zeros(dataU.shape[1])
    for i in tqdm(range(dataU.shape[1]), desc='GetEnergy'):
        u = dataU[:, i]
        E[i] = np.real(np.trapz(u * np.conj(u), t))
    return E * P0

def GetPulseByEnergyRange(dataset, energy_low, energy_high):
    E = GetEnergy(dataset.u)
    return dataset.u[:, (E >= energy_low) & (E < energy_high)]

def GetPulseByEnergyLogRange(dataset, energy_low, energy_high):
    E = np.log10(GetEnergy(dataset.u))
    return dataset.u[:, (E >= energy_low) & (E < energy_high)]

def ScanEnergyRange(dataset, nrange=100):
    E = GetEnergy(dataset.u)
    Emin, Emax = np.min(E), np.max(E)
    dE = (Emax - Emin) / nrange
    nE = []
    E1 = Emin
    for ii in tqdm(range(nrange), desc='ScanEnergyRange'):
        E2 = E1 + dE
        nE.append(len(E[(E >= E1) & (E < E2)]))
        E1 = E2
    return np.array(nE)

def ScanEnergyRangeLog(dataset, nrange=100):
    E = np.log10(GetEnergy(dataset.u))
    Emin, Emax = np.min(E), np.max(E)
    dE = (Emax - Emin) / nrange
    print(Emin, Emax, dE)
    nE = []
    E1 = Emin
    for ii in tqdm(range(nrange), desc='ScanEnergyRangeLog'):
        E2 = E1 + dE
        nE.append(len(E[(E >= E1) & (E < E2)]))
        E1 = E2
    return np.array(nE)

def SeperatePulseByEnergyRanges(dataset, nrange):
    e = GetEnergy(dataset.u)
    Emin, Emax = np.min(e), np.max(e)
    dE = (Emax - Emin) / nrange
    U = []
    Q = []
    E = []
    E1 = Emin
    for ii in tqdm(range(nrange), desc='SeperatePulseByEnergyLogRanges'):
        E2 = E1 + dE
        U.append(dataset.u[:, (e >= E1) & (e < E2)])
        Q.append(dataset.q[:, (e >= E1) & (e < E2)])
        E.append(e[(e >= E1) & (e < E2)])
        E1 = E2
    return U, Q, (E, Emin, Emax, dE)

def SeperatePulseByEnergyLogRanges(dataset, nrange):
    e = np.log10(GetEnergy(dataset.u))
    Emin, Emax = np.min(e), np.max(e)
    dE = (Emax - Emin) / nrange
    U = []
    Q = []
    E = []
    E1 = Emin
    for ii in tqdm(range(nrange), desc='SeperatePulseByEnergyLogRanges'):
        E2 = E1 + dE
        U.append(dataset.u[:, (e >= E1) & (e < E2)])
        Q.append(dataset.q[:, (e >= E1) & (e < E2)])
        E.append(e[(e >= E1) & (e < E2)])
        E1 = E2
    return U, Q, (E, Emin, Emax, dE)

def SeperatePulseByEnergyRanges_(u, q, nrange):
    e = GetEnergy(u)
    Emin, Emax = np.min(e), np.max(e)
    dE = (Emax - Emin) / nrange
    U = []
    Q = []
    E = []
    E1 = Emin
    for ii in tqdm(range(nrange), desc='SeperatePulseByEnergyRanges'):
        E2 = E1 + dE
        U.append(u[:, (e >= E1) & (e < E2)])
        Q.append(q[:, (e >= E1) & (e < E2)])
        E.append(e[(e >= E1) & (e < E2)])
        E1 = E2
    return U, Q, (E, Emin, Emax, dE)

def SeperatePulseByEnergyLogRanges_(u, q, nrange):
    e = np.log10(GetEnergy(u))
    Emin, Emax = np.min(e), np.max(e)
    dE = (Emax - Emin) / nrange
    U = []
    Q = []
    E = []
    E1 = Emin
    for ii in tqdm(range(nrange), desc='SeperatePulseByEnergyLogRanges'):
        E2 = E1 + dE
        U.append(u[:, (e >= E1) & (e < E2)])
        Q.append(q[:, (e >= E1) & (e < E2)])
        E.append(e[(e >= E1) & (e < E2)])
        E1 = E2
    return U, Q, (E, Emin, Emax, dE)

def prepareQ(q, normalization = True):
    qtemp = q
    qnorm = 1.0
    if normalization:
        qnorm = np.max(np.abs(qtemp))
        qtemp /= qnorm
    qreal = np.real(qtemp)
    qimag = np.imag(qtemp)
    Q = torch.tensor(np.c_[qreal,qimag].T)
    Q = Q.reshape((1,) + Q.shape)
    return Q, qnorm

def prepareU(u, normalization = True):
    phaseComp = np.exp(1j*np.pi*np.arange(len(u))) * 2e5
    utemp = -np.conj(np.complex64(np.fft.fftshift(np.fft.ifft(u)) * phaseComp))
    return prepareQ(utemp, normalization)

def valid_(model, u, q, loss_fn, transformU, transformQ, normalization = True, forward=False):
    model.eval()
    device = next(model.parameters()).device
    losses = np.zeros(u.shape[1])
    with torch.no_grad():
        for ii in tqdm(range(u.shape[1]), desc='Valid'):
            x, _ = transformU(u[:, ii], normalization)
            y, _ = transformQ(q[:, ii], normalization)
            x, y = x.to(device), y.to(device)
            if forward:
                y, x = x, y
            with torch.cuda.amp.autocast():                
                y_hat = model(x)
                losses[ii] = loss_fn(y_hat, y).detach().cpu().numpy()
        loss = np.mean(losses)
    return loss, losses

def valid(model, dataset, loss_fn, batch_size, forward=False):
    model.eval()
    device = next(model.parameters()).device
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses = []
    pbar = tqdm(dataloader, disable=False)
    with torch.no_grad():
        for x, y, _ in pbar:
            pbar.set_description("Valid")
            x, y = x.to(device), y.to(device)
            if forward:
                y, x = x, y
            y_hat = model(x)
            losses.append(loss_fn(y_hat, y).detach().cpu().numpy())
        losses = np.concatenate(losses)
        loss = np.sqrt(np.mean(losses**2))
    return loss, losses

def valid2(model1, model2, dataset, loss_fn, batch_size, forward=False):
    model1.eval()
    model2.eval()
    device = next(model1.parameters()).device
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses = []
    pbar = tqdm(dataloader, disable=False)
    for y, x, _ in pbar:
        pbar.set_description("Valid")
        x, y = x.to(device), y.to(device)
        if forward:
            y, x = x, y
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                x_hat = model1(x)
                y_hat = model2(x_hat)
                losses.append(loss_fn(y_hat, y).detach().cpu().numpy())
    losses = np.concatenate(losses)
    loss = np.sqrt(np.mean(losses**2))
    return loss, losses

def LossVsEnergy_(model, u, q, loss_fn, nrange=100):
    U, Q, E = SeperatePulseByEnergyRanges_(u, q, nrange)
    loss = []
    for ii in tqdm(range(len(U)), desc='LossVsEnergy'):
        loss.append( valid(model, U[ii], Q[ii], loss_fn, prepareU, prepareQ)[0] )
    return loss, E

def LossVsEnergy(model, dataset, loss_fn, nrange=100):
    U, Q, E = SeperatePulseByEnergyRanges(dataset, nrange)
    loss = []
    for ii in tqdm(range(len(U)), desc='LossVsEnergy'):
        loss.append( valid(model, U[ii], Q[ii], loss_fn, prepareU, prepareQ)[0] )
    return loss, E

def LossVsEnergyLog_(model, u, q, loss_fn, nrange=100):
    U, Q, E = SeperatePulseByEnergyLogRanges_(u, q, nrange)
    loss = []
    for ii in tqdm(range(len(U)), desc='LossVsEnergyLog'):
        loss.append( valid(model, U[ii], Q[ii], loss_fn, prepareU, prepareQ)[0] )
    return loss, E

def LossVsEnergyLog(model, dataset, loss_fn, nrange=100):
    U, Q, E = SeperatePulseByEnergyLogRanges(dataset, nrange)
    loss = []
    for ii in tqdm(range(len(U)), desc='LossVsEnergyLog'):
        loss.append( valid(model, U[ii], Q[ii], loss_fn, prepareU, prepareQ)[0] )
    return loss, E
