import numpy as np

###### function approximators ######### 
class functions():
    def __init__(self,**args):
        temp = 0
        self.approximator = 0
        if args['approximator'] == 'linear':
            self.approximator = self.linear_approximator
        if args['approximator'] == 'fourier':
            self.approximator = self.fourier_approximator           
        
            
    ### -- linear approximator -- ###
    def linear_approximator(self,W):
        # loop over weights and update
        N,M = np.shape(W)
        y = 0
        for n in range(N):
            w = W[n,:]
            w0 = w[0]
            w1 = w[1:]
            y += w0 + np.dot(w1[1:],s)
        return y

    ### -- cosine approximator -- ###
    def cosine_approximator(self,W):                 # Define a function
        N,M = np.shape(W)
        y = 0
        c = 1
        for n in range(N):
            w = W[n,:]
            w0 = w[0]
            w1 = w[1:]
            y += w0 + np.dot(w1,np.cos(s))
        return y
    