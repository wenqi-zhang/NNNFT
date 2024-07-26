import torch
import torch.nn as nn
from NFTModel import ConvLSTMAutoEncoder
import numpy as np
from utils import plotresult_, prepareU
from scipy.io import savemat

mfname='modelnfttrain9.pth'

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

#########################################################################################

nt = 2048
nc = 2

model = ConvLSTMAutoEncoder(nc, nc, nt, kernel_size=3, features=[64, 128, 256], lstm_layers=1, forward=False).to(device)

#########################################################################################

def loss_fn(input, target):
    return torch.sqrt(nn.functional.mse_loss(input, target))

print('Load previously saved model: ' + mfname + ' ...')
savedata = torch.load(mfname, map_location=device)
model.load_state_dict(savedata['model'], strict=False)
model.eval()

nt = 2**11
dt = 1.0 / 96e9
T = nt * dt
T0 = np.sqrt(np.abs(-2.172e-26) / 2.0)
TT = T / T0
t = np.arange(nt) / nt * TT - TT / 2.0
A = 0.4
q = A / 250 / np.cosh(t / 250)
x, qlam = plotresult_(model, q)

savedict = {
    't' : t,
    'x' : x,
    'qlam' : qlam
}

savemat("nftinferencesech.mat", savedict)

A = 0.4
q = np.zeros_like(t);
q[abs(t) < 2000] = A / 2000
x, qlam = plotresult_(model, q)

savedict = {
    't' : t,
    'x' : x,
    'qlam' : qlam
}

savemat("nftinferencerect.mat", savedict)

A = 1
q = A / 1500 * np.sinc(t / 1500) * np.exp(1j * np.pi)
x, qlam = plotresult_(model, q)

savedict = {
    't' : t,
    'x' : x,
    'qlam' : qlam
}

savemat("nftinferencesinc.mat", savedict)

# traced_script_module = torch.jit.trace(model, prepareU(q)[0].cuda())
# traced_script_module.save("traced_model.pt")
