# import basic data loading and handling library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# import sklearn libraries 
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor

# import widgets and animators

# load in other libs
import sys
sys.path.append('../')

# import JS animator
import matplotlib.animation as animation
from JSAnimation_slider_only import IPython_display


class Fit_Bases:
    
    def __init__(self):
        self.x = 0
        self.y = 0
        self.target_x = []
        self.target_y = []

    # load in data
    def load_data(self,csvname):
        data = np.asarray(pd.read_csv(csvname,header = None))
        self.x = data[:,0][:, np.newaxis]
        self.y = data[:,1]
        
    # load target function
    def load_target(self,csvname):
        data = np.asarray(pd.read_csv(csvname,header = None))
        self.target_x = data[:,0][:, np.newaxis]
        self.target_y = data[:,1]
        
    # plot data with underlying target function
    def plot_all(self,ax):
        # plot target if loaded
        if len(self.target_x) > 1:
            ax.plot(self.target_x,self.target_y,'r--',linewidth = 2.5,zorder = 0)
        
        # plot data if loaded
        if len(self.x) > 1:
            ax.scatter(self.x,self.y,facecolor = 'k',edgecolor = 'w',linewidth = 1,s = 70)

        ax.set_xlim(min(self.x)-0.1,max(self.x)+0.1)
        ax.set_ylim(min(self.y)-0.1,max(self.y)+0.1)
        ax.set_yticks([],[])
        ax.axis('off')      

    ### demo with animation or sliders - showing function approximation with polynoimal, neural network, and stumps/trees
    # polys
    def browse_poly_fit(self):
        
        # initialize figure
        fig = plt.figure(figsize = (5,4))
        artist = fig
        ax = fig.add_subplot(111)
        r = np.linspace(min(self.x),max(self.x),300)[:, np.newaxis]

        def show_fit(num_elements):
            ax.cla()

            # plot our points and target function
            self.plot_all(ax)
            
            # define classifier object
            clf = KernelRidge(kernel = 'poly',degree = num_elements)
            clf.fit(self.x, self.y)        
            
            # clf = GradientBoostingRegressor(n_estimators=num_elements, learning_rate=1,max_depth=2, random_state=0, loss='ls')
            
            # clf = MLPRegressor(solver = 'lbgfs',alpha = 0,activation = 'tanh',random_state = 1,hidden_layer_sizes = (num_elements,num_elements))
                
            # plot approximation
            z = clf.predict(r)

            # plot regressor
            ax.plot(r,z,linewidth = 3,color = 'b')
            ax.set_ylim(min(min(self.y)-0.1,min(z)-0.1),max(max(self.y)+0.1,max(z)+0.1))
            return artist,
           
        anim = animation.FuncAnimation(fig, show_fit,frames=20, interval=20, blit=True)
        
        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = 20)
        
        return(anim)
       
