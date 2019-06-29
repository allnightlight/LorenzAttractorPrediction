
import torch
import torch.nn as nn
import numpy as np


class Predictor(nn.Module):
    def __init__(self, Nx, Ny, Nlayers):
        super(Predictor, self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.Nlayers = Nlayers

        self.q_x_y = nn.Sequential(nn.Linear(Ny*Nlayers, Nx*Nlayers), 
            nn.Tanh(),)
        self.p_y_x = nn.Sequential(nn.Linear(Nx, Nx), nn.Tanh(),
            nn.Linear(Nx, Ny),)
        self.f_x_xw = nn.GRU(1, Nx, Nlayers) 

    def forward(self, _Y0, Nseq):
# _Y: (Nlayers, *, Ny)
        Nx, Ny, Nlayers = self.Nx, self.Ny, self.Nlayers

        Nbatch = _Y0.shape[1]

        _x0 = self.q_x_y(_Y0.transpose(0,1).reshape(Nbatch, 
            Nlayers * Ny)).reshape(Nbatch, Nlayers, Nx).transpose(0, 1)
# _x0: (Nlayers, *, Nx)
        _U = torch.zeros(Nseq, Nbatch, 1) # (Nseq, *, 1)
        _X, _  = self.f_x_xw(_U, _x0) # _X: (Nseq, *, Nx)
        _Yhat = self.p_y_x(_X) # (Nseq, *, Ny)

        return _Yhat
        

#Nx, Ny, Nlayers = 2**4, 2**2, 2**1
#
#model = Predictor(Nx, Ny, Nlayers)
#Nbatch = 2**0
#N = 2**3
#_Y = torch.randn(Nlayers, Nbatch, Ny)
#
#_Yhat = model(_Y, N)
#print(_Yhat.shape)
