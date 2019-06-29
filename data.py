from scipy.integrate import odeint
import numpy as np

class GeneratorFromLorenzAttractor():
    def __init__(self, T = 1000, dt = 0.1, test_ratio = 0.1):
        def f(x, t):
            p,r,b = 10,28,8/3
            dxdt = [
                -p * x[0] + p * x[1],
                -x[0] * x[2] + r * x[0] - x[1],
                x[0] * x[1] - b * x[2],
            ]
            return dxdt

        t = np.arange(0, T, dt)
        x0 = [-6.12, -5.16, 25.6]
        X = odeint(f, x0, t).astype(np.float32)
        self.X = ( X - np.mean(X, axis=0) ) / np.std(X, axis=0)
        self.Ntest = int(len(t) * test_ratio)
        self.Ntrain = len(t) - self.Ntest
        self.Nx = 3

    def batch(self, Nbatch, N):
        idx = np.random.randint(low=0, high=self.Ntrain-N, size=(1,Nbatch)) + \
            np.arange(N).reshape(-1,1) # (N, Nbatch)
        Xbatch = self.X[idx,:] # (N, Nbatch, 3)
        return Xbatch

    def test(self, N):
        Nbatch = self.Ntest - N
        idx = np.random.randint(low=self.Ntrain, high=self.Ntrain+self.Ntest-N, 
            size=(1,Nbatch)) + np.arange(N).reshape(-1,1) # (N, Nbatch)
        Xbatch = self.X[idx,:] # (N, Nbatch, 3)
        return Xbatch, Nbatch
