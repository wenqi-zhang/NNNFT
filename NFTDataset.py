import torch
from torch.utils.data import Dataset
import numpy as np
import mat73
import gc

class NFTDataset(Dataset):
    def __init__(self, matfile, test=0, testportion=0, lengthlimit=None, normalization=False, timedomain=False, transform=None):
        super(NFTDataset, self).__init__()
        print( 'Preload all data into memory ... ')
        try:
            f = open(matfile+".npz","rb")
            npz = np.load(f)
            t = npz['t']
            lam = npz['lam']
            UTime = npz['UTime']
            QLambda = npz['QLambda']
            preShiftL = npz['preShiftL']
            PulseT0 = npz['PulseT0']
            Label = npz['Label']
            inch = npz['inch']
            inAA = npz['inAA']
            cf = npz['cf']
            f.close()
            del npz
        except:
            mat = mat73.loadmat(matfile)
            t = np.array(mat['t'],dtype=np.float32)
            lam = np.array(mat['lam'],dtype=np.float32)
            UTime = np.array(mat['UTime'],dtype=np.complex64)
            QLambda = np.array(mat['QLambda'],dtype=np.complex64)
            preShiftL = np.array(mat['preShiftL'],dtype=np.float32)
            PulseT0 = np.array(mat['PulseT0'],dtype=np.float32)
            Label = np.array(mat['Label'],dtype=np.uint8)
            inch = np.array(mat['inch'],dtype=np.uint8)
            inAA = np.array(mat['inAA'],dtype=np.uint8)
            cf = np.array(mat['carrierfunc'],dtype=np.uint8)
            del mat
            f = open(matfile+".npz","wb")
            np.savez(f,t=t,lam=lam,UTime=UTime,QLambda=QLambda,Label=Label,inAA=inAA,inch=inch,preShiftL=preShiftL,PulseT0=PulseT0,cf=cf)
            f.close()

        if lengthlimit:
            self.len = lengthlimit
            UTime = UTime[:,:self.len]
            QLambda = QLambda[:,:self.len]
            preShiftL = preShiftL[:self.len]
            PulseT0 = PulseT0[:self.len]
            Label = Label[:,:self.len]
            inch = inch[:self.len]
            inAA = inAA[:self.len]
            cf = cf[:self.len]
        
        self.normalization = normalization
        self.len = Label.shape[1]

        self.transform = transform
        self.timedomain = timedomain

        if test==0:
            self.len = self.len-int(self.len*testportion)
            UTime = UTime[:,:self.len]
            QLambda = QLambda[:,:self.len]
            preShiftL = preShiftL[:self.len]
            PulseT0 = PulseT0[:self.len]
            Label = Label[:,:self.len]
            inch = inch[:self.len]
            inAA = inAA[:self.len]
            cf = cf[:self.len]
        else:
            self.len = int(self.len*testportion)
            UTime = UTime[:,-self.len:]
            QLambda = QLambda[:,-self.len:]
            preShiftL = preShiftL[-self.len:]
            PulseT0 = PulseT0[-self.len:]
            Label = Label[:,-self.len:]
            inch = inch[-self.len:]
            inAA = inAA[-self.len:]
            cf = cf[-self.len:]

        self.t = t.copy()
        self.lam = lam.copy()
        self.u = UTime.copy()
        self.q = QLambda.copy()
        self.psl = preShiftL.copy()
        self.t0 = PulseT0.copy()
        self.l = Label.copy() - 1
        self.ch = inch - 1
        self.aa = inAA - 1
        self.cf = cf
        
        self.phaseComp = np.exp(1j*np.pi*np.arange(len(self.u[:,0]))) * 2e5
        del t
        del lam
        del UTime
        del QLambda
        del preShiftL
        del PulseT0
        del Label
        del inch
        del inAA
        del cf

        gc.collect()
       
    def __delete__(self):
        pass

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if type(self.timedomain) == bool:
            if self.timedomain:
                utemp = self.u[:,idx]
            else:
                utemp = -np.conj(np.complex64(np.fft.fftshift(np.fft.ifft(self.u[:,idx])) * self.phaseComp))
        elif type(self.timedomain) == str:
            if 'U' in self.timedomain.upper():
                utemp = self.u[:,idx]
            else:
                utemp = -np.conj(np.complex64(np.fft.fftshift(np.fft.ifft(self.u[:,idx])) * self.phaseComp))
        
        if type(self.normalization) == bool:
            if self.normalization:
                unorm = np.max(np.abs(utemp))
                utemp /= unorm
        elif type(self.normalization) == str:
            if 'U' in self.normalization.upper():
                unorm = np.max(np.abs(utemp))
                utemp /= unorm
        ureal = np.real(utemp)
        uimag = np.imag(utemp)
        
        nt = self.t.shape[0]
        
        if type(self.timedomain) == bool:
            if self.timedomain:
                qtemp = np.complex64(np.fft.fft(np.fft.fftshift(np.conj(-self.q[:,idx] / self.phaseComp)))) * nt
            else:
                qtemp = self.q[:,idx]
        elif type(self.timedomain) == str:
            if 'Q' in self.timedomain.upper():
                qtemp = np.complex64(np.fft.fft(np.fft.fftshift(np.conj(-self.q[:,idx] / self.phaseComp)))) * nt
            else:
                qtemp = self.q[:,idx]
        
        if type(self.normalization) == bool:
            if self.normalization:
                qnorm = np.max(np.abs(qtemp))
                qtemp /= qnorm
        elif type(self.normalization) == str:
            if 'Q' in self.normalization.upper():
                qnorm = np.max(np.abs(qtemp))
                qtemp /= qnorm
        qreal = np.real(qtemp)
        qimag = np.imag(qtemp)

        l = self.l[:,idx]
        u = np.c_[ureal,uimag].T
        q = np.c_[qreal,qimag].T

        if self.transform:
            u = self.transform(u)
            q = self.transform(q)
            l = self.transform(l)

        return u, q, l
