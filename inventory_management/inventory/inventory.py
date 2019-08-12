import numpy as np

class Forecaster:
    def __init__(self, c, m, sigma):
        self.c     = c
        self.m     = m
        self.sigma = sigma

    def predict(self, Y0, T):
        Yhat = np.zeros((T+self.m))
        Yhat[:self.m] = Y0

        for i in range(self.m,T+self.m):
            Yhat[i] = self.c + Yhat[i-self.m] + self.sigma*np.random.randn()

        return Yhat[self.m:]

def sim(x0, y0, U, A, B, C, T, forecaster):
    n = x0.shape[0]
    X = np.zeros((n, T))
    X[:,0] = x0
    S = np.zeros((T))

    Y = forecaster.predict(y0, T)

    for i in range(1,T):
        y = Y[i-1]

        if y > X[0,i-1]:
            S[i-1] = X[0,i-1]
        else:
            S[i-1] = y

        X[:,i] = A.dot(X[:,i-1]) + B.dot(U[i-1]) + C.dot(S[i-1])

    return X,Y,S
