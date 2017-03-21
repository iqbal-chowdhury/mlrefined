# import basic data loading and handling library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import copy

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

# begin class
class RegressionFit:
    
    def __init__(self):
        self.x = 0
        self.y = 0
        self.target_x = 0
        self.target_y = 0
    
    ### data loading functions ###
    # load target function
    def load_target(self,targetcsv):
        data = np.asarray(pd.read_csv(targetcsv))
        self.target_x = data[:,0:-1]
        self.target_y = data[:,-1]
        
    # load dataset
    def load_data(self,datacsv):
        # load data
        data = np.asarray(pd.read_csv(datacsv))  
        self.x = data[:,0:-1]
        self.y = data[:,-1]
        
    ### model selection and prediction function generation ###
    def model_selection(self,model_choice,value):
        z = 0
        # polynomial
        if model_choice == 'poly':
            # get an instance of the linear regressor + transform input via poly
            poly = PolynomialFeatures(degree=value)
            x = poly.fit_transform(self.x)
            regressor = linear_model.LinearRegression()

            # fit your model to data
            regressor.fit(x, self.y)   

            # make prediction
            pr = poly.fit_transform(self.target_x)

            # produce approximation
            z = regressor.predict(pr)
                
        # neural network
        if model_choice == 'nnet':
            regressor = MLPRegressor(solver = 'lbgfs',alpha = 0,activation = 'tanh',max_iter = 500,hidden_layer_sizes = (value,value,value,value),tol=10**-5)

            # fit your model to data
            regressor.fit(self.x, self.y)   

            # produce approximation
            z = regressor.predict(self.target_x)
                
        # tree-based
        if model_choice == 'tree':
            regressor = GradientBoostingRegressor(n_estimators= value, learning_rate=1,max_depth=3, random_state=0, loss='ls')

            # fit your model to data
            regressor.fit(self.x, self.y)   

            # produce approximation
            z = regressor.predict(self.target_x)
                
        # return approximation
        return z
    
    
    ### plotting functions ###
    # plot data with underlying target function
    def show_setup(self,ax):
        # plot target if loaded 
        if self.target_x.ndim > 1:
            # check dimension of plot
            if np.shape(self.target_x)[1] > 1:
                if np.shape(self.target_x)[1] == 1:  # two-dimensional target function
                    ax.plot(self.target_x,self.target_y,'r--',linewidth = 2.5,zorder = 0)
                else:                              # three-dimensional target function
                    a = int(math.sqrt(len(self.target_x[:,0])))
                    ax.plot_surface(np.reshape(self.target_x[:,0],(a,a)),np.reshape(self.target_x[:,1],(a,a)),np.reshape(self.target_y,(a,a)),alpha = 0.05,color = 'r',zorder = 0,shade = True,linewidth = 0)

        ### plot data ###
        # check dimension of plot
        if np.shape(self.x)[1] == 1:      # two-dimensional target function
            ax.scatter(self.x,self.y,facecolor = 'k',edgecolor = 'w',linewidth = 1,s = 70)
        else:                             # three-dimensional target function
            ax.scatter(self.x[:,0],self.x[:,1],self.y,s = 50,color = 'k',edgecolor = 'w')

        ### dress panel ###
        # check dimension of plot 
        if np.shape(self.x)[1] == 1:      # two-dimensional target function
            self.panel_2d_cleanup(ax)
        else:                             # three-dimensional target function
            self.panel_3d_cleanup(ax)


    # function for cleaning up 2-dim panel in a consistent way
    def panel_2d_cleanup(self,ax):
        ax.set_yticks([],[])
        ax.axis('off')     
        
        # set viewing lims
        xgap = (max(self.x) - min(self.x))*0.1
        ygap = (max(self.y) - min(self.y))*0.15
        ax.set_xlim(min(self.x) - xgap, max(self.x) + xgap)
        ax.set_ylim(min(self.y) - ygap, max(self.y) + ygap)
        
    # function for cleaning up 3-dim panel in a consistent way
    def panel_3d_cleanup(self,ax):
        # turn off tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Get rid of the spines on the 3d plot
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # turn off tick marks
        ax.xaxis.set_tick_params(size=0,color = 'w')
        ax.yaxis.set_tick_params(size=0,color = 'w')
        ax.zaxis.set_tick_params(size=0,color = 'w')

        # set lims
        x1gap = (max(self.x[:,0]) - min(self.x[:,0]))*0.1
        x2gap = (max(self.x[:,1]) - min(self.x[:,1]))*0.1
        ygap = (max(self.y) - min(self.y))*0.15
        
        ax.set_xlim([min(self.x[:,0])-x1gap,max(self.x[:,0])+x1gap])
        ax.set_ylim([min(self.x[:,1])-x2gap,max(self.x[:,1])+x2gap])
        ax.set_zlim([min(self.y)-ygap,max(self.y)+ygap])
            
        # set viewing angle
        ax.view_init(10,-70) 
        
    ### animation function ###
    def browse_fit(self,**args):
        # pull out model arg
        model_choice = args['model']
        if model_choice not in ['poly','nnet','tree']:
            print 'please try again - choose from: poly, nnet, or tree'
            return
        
        # pull out range
        param_range = args['param_range']
        
        # initialize figure
        fig = plt.figure(figsize = (7,7))
        artist = fig
        
        # switch - two-d or three-d plot
        ax = 0
        if np.shape(self.x)[1] == 1:     # two-dimensional plot
            # seed panel
            ax = fig.add_subplot(111)
        else:                            # three-dimensional plot
            ax = plt.subplot(111,projection = '3d')
        
        # remove whitespace around figure
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1) 
        
        # check to see if target loaded, if not create target range for prediction functions
        if np.prod(np.shape(self.target_x)) == 1:
            xgap = (max(self.x) - min(self.x))*0.05
            self.target_x = np.linspace(min(self.x) - xgap, max(self.x) + xgap,200)
            self.target_x.shape = (len(self.target_x),1)
        
        # animation sub-function
        def show_fit(value):
            ax.cla()

            # plot our points and target function
            self.show_setup(ax)
            
            # fit model to data, then use to predict over the region of input of the true function to make approximation
            z = self.model_selection(model_choice,param_range[value])
           
            # plot regressor - two-d or three-d - and clean up plot
            if np.shape(self.x)[1] == 1:     # two-dimensional plot
                ax.plot(self.target_x,z,linewidth = 3,color = 'b')
            else:                            # three-dimensional plot
                a = int(math.sqrt(len(self.target_x[:,0])))
                ax.plot_surface(np.reshape(self.target_x[:,0],(a,a)),np.reshape(self.target_x[:,1],(a,a)),np.reshape(z,(a,a)),alpha = 0.1,color = 'b',zorder = 0,shade = True,linewidth=0.5,antialiased = True,cstride = 50, rstride = 50)
            
            # dress up panel title correctly
            if value == 0:
                ax.set_title(model_choice + ' fit with ' + str(param_range[value]) + ' basis element',y=0.9,fontsize = 15)
            else:
                ax.set_title(model_choice + ' fit with ' + str(param_range[value]) + ' basis elements',y=0.9,fontsize = 15)
               
            return artist,
           
        anim = animation.FuncAnimation(fig, show_fit,frames=len(param_range), interval=len(param_range), blit=True)
        
        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = len(param_range))
        
        return(anim)
       
