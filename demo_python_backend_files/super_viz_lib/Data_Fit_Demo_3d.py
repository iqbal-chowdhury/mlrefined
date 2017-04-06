# import basic data loading and handling library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# import sklearn libraries 
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
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
class two_dim_regression_fits:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.target_x = 0
        self.target_y = 0
    
    # load target function
    def load_target(self,targetcsv):
        data = np.asarray(pd.read_csv(csvname,header = None))
        self.target_x = data[:,-1]
        self.target_y = data[:,-1]
        
    # load dataset
    def load_data(self,datacsv):
        # load data
        data = np.asarray(pd.read_csv(datacsv))  
        self.x = data[:,0:-1]
        self.y = data[:,-1]
        
    # plot the data and target
    def plot_basics(self):
        # convert globals
        x = self.x
        y = self.y 
        s = self.x_surf 
        t =  self.y_surf 
        z_surf = self.z_surf
        
        # create input for classifier surface fit
        s_copy = np.asarray(s.copy())
        s_copy.shape = (np.prod(np.shape(s_copy)),1)
        t_copy = np.asarray(t.copy())
        t_copy.shape = (np.prod(np.size(t_copy)),1)
        h = np.concatenate((s_copy,t_copy),axis = 1)
        
        #### plot everything ###
        # produce figure
        fig = plt.figure(num=None, figsize=(5,5), dpi=80, facecolor='w', edgecolor='k')
        ax1 = plt.subplot(111,projection = '3d')
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)   # remove whitespace around 3d figure
        
        ax1.cla()
        ax1.axis('tight')

        ### plot all input data ###
        # plot points
        ax1.scatter(X[:,0],X[:,1],y,s = 50,color = 'k',edgecolor = 'w')

        # plot target surface
        ax1.plot_surface(s,t,z_surf,alpha = 0.05,color = 'r',zorder = 0,shade = True,linewidth = 0)

        # turn off tick labels
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])

        # Get rid of the spines on the 3d plot
        ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # turn off tick marks
        ax1.xaxis.set_tick_params(size=0,color = 'w')
        ax1.yaxis.set_tick_params(size=0,color = 'w')
        ax1.zaxis.set_tick_params(size=0,color = 'w')

        # set limits
        ax1.set_xlim([-0.05,1.05])
        ax1.set_ylim([-0.05,1.05])
        ax1.set_zlim([-1.1,1.1])

        # set viewing angle
        ax1.view_init(10,-70)        
        
    # browse poly fit values
    def browse_fit(self,**args):
        # pull out model arg
        model_choice = args['model']
        if model_choice not in ['poly','nnet','tree']:
            print 'please try again - choose from: poly, nnet, or tree'
            return
        
        # pull out range
        param_range = args['param_range']
        
        # convert globals
        x = self.x
        y = self.y 
        s = self.x_surf 
        t =  self.y_surf 
        z_surf = self.z_surf
        
        # create input for regression surface fit
        
        s_copy = np.asarray(s.copy())
        s_copy.shape = (np.prod(np.shape(s_copy)),1)
        t_copy = np.asarray(t.copy())
        t_copy.shape = (np.prod(np.size(t_copy)),1)
        r = np.concatenate((s_copy,t_copy),axis = 1)
        
        #### plot everything ###
        # produce figure
        fig = plt.figure(num=None, figsize=(5,5), dpi=80, facecolor='w', edgecolor='k')
        ax1 = plt.subplot(111,projection = '3d')
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)   # remove whitespace around 3d figure
        
        def show_fit(value):
            ax1.cla()
            ax1.axis('tight')
            
            ### plot all input data ###
            # plot points
            ax1.scatter(self.x[:,0],self.x[:,1],self.y,s = 50,color = 'k',edgecolor = 'w')

            # plot target surface
            artist = ax1.plot_surface(s,t,z_surf,alpha = 0.05,color = 'r',zorder = 0,shade = True,linewidth = 0)

            ### create regressor and fit to data ###
            ### choose between models
            # polynomial
            if model_choice == 'poly':
                poly = PolynomialFeatures(degree=value+1)
                x = poly.fit_transform(self.x)
                regressor = linear_model.LinearRegression()
                
                # fit your model to data
                regressor.fit(x, self.y)   
                
                # make prediction
                pr = poly.fit_transform(r)
                
                # produce approximation
                z = regressor.predict(pr)
                
            # neural network
            if model_choice == 'nnet':
                regressor = MLPRegressor(solver = 'lbgfs',alpha = 0,activation = 'tanh',max_iter = 500,hidden_layer_sizes = (param_range[value],param_range[value],param_range[value],param_range[value]),tol=10**-5)
                
                # fit your model to data
                regressor.fit(self.x, self.y)   
            
                # produce approximation
                z = regressor.predict(r)
                
            # tree-based
            if model_choice == 'tree':
                regressor = GradientBoostingRegressor(n_estimators= param_range[value], learning_rate=1,max_depth=1, random_state=0, loss='ls')
             
                # fit your model to data
                regressor.fit(self.x, self.y)   

                # produce approximation
                z = regressor.predict(r)
            
            ### plot surface of prediction
            z.shape = (np.shape(s))
            
            ### plot regression surface ###
            ax1.plot_surface(s,t,z,alpha = 0.1,color = 'b',zorder = 0,shade = True,linewidth=0.5,antialiased = True,cstride = 50, rstride = 50)

            ### clean up plot ###
            ax1.set_title('tree fit with ' + str(value+1) + ' basis elements')
            
            # turn off tick labels
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_zticklabels([])

            # Get rid of the spines on the 3d plot
            ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            # turn off tick marks
            ax1.xaxis.set_tick_params(size=0,color = 'w')
            ax1.yaxis.set_tick_params(size=0,color = 'w')
            ax1.zaxis.set_tick_params(size=0,color = 'w')
            
            # set limits
            ax1.set_xlim([-0.05,1.05])
            ax1.set_ylim([-0.05,1.05])
            ax1.set_zlim([-1.1,1.1])

            # set viewing angle
            ax1.view_init(10,-70)        

            return artist,
        
        anim = animation.FuncAnimation(fig, show_fit,frames=len(param_range), interval=len(param_range), blit=True)
        
        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = len(param_range))
        
        return(anim)
 