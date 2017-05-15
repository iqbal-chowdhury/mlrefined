###### function approximators ######### 
class function_approximators():
    ### -- linear approximator -- ###
    def linear_approximator(self,W):
        y = np.dot(W.T,s)
        return y

    ### -- fourier approximator -- ###
    def fourier_approximator(self,W):                 # Define a function
        N,M = np.shape(W)
        y = 0
        c = 1
        for n in range(N):
            print (n)
            w = W[n,:]
            w0 = w[0]
            w1 = w[1:]
            temp1 = 0

            # make summand
            if np.mod(n,2) == 0:
                temp1 = w0*np.sin(c*(w1[0] + np.dot(w1[1:],s)))
                print ('got here')
            else:
                temp1 = w0*np.cos(c*(w1[0] + np.dot(w1[1:],s)))
                c += 1
                print ('got there')
            y += temp1
        return y