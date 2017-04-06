import numpy as np
import pandas as pd

class learner():
    def __init__(self,**args):
        # get some crucial parameters from the input gridworld
        self.grid = args['gridworld']
        
        # initialize q-learning params
        self.gamma = 1
        self.max_steps = 5*self.grid.width*self.grid.height
        self.exploit_param = 0.5
        self.step_size = 10**-2
        self.action_method = 'exploit'
        self.training_episodes = 500
        self.validation_episodes = 50
        self.training_start_schedule = []
        self.validation_start_schedule = []
        
        # swap out for user defined q-learning params if desired
        if "gamma" in args:
            self.gamma = args['gamma']
        if 'max_steps' in args:
            self.max_steps = args['max_steps']
        if 'action_method' in args:
            self.action_method = args['action_method']
        if 'step_size' in args:
            self.step_size = args['step_size']
        if 'exploit_param' in args:
            self.exploit = args['exploit_param']
            self.action_method = 'exploit'
        if 'training_episodes' in args:
            self.training_episodes = args['training_episodes']
            # return error if number of training episodes is too big
        if self.training_episodes > self.grid.training_episodes:
            print 'requesting too many training episodes, the maximum num = ' + str(self.grid.training_episodes)
            return 
        self.training_start_schedule = self.grid.training_start_schedule[:self.training_episodes]       
        if 'validation_episodes' in args:
            self.validation_episodes = args['validation_episodes']
            # return error if number of training episodes is too big
        if self.validation_episodes > self.grid.validation_episodes:
            print 'requesting too many validation episodes, the maximum num = ' + str(self.grid.validation_episodes)
            return 
        self.validation_start_schedule = self.grid.validation_start_schedule[:self.validation_episodes]
        
        # initialize function approximation params and weights
        self.deg = 1
        if 'degree' in args:
            self.deg = args['degree']
            
        # count dimension of poly
        test_output = self.poly_features(state = np.zeros((1,2))[0])
        self.W = np.random.randn(len(test_output),4)
        
    # builds (poly) features based on input data 
    def poly_features(self,state):
        # normalize state data
        state[0]=state[0]/float(self.grid.width)
        state[1]=state[1]/float(self.grid.height)

        # produce polynomials of normalized state data
        F = []
        for n in range(self.deg+1):
            for m in range(self.deg+1):
                if n + m <= self.deg:
                    temp = (state[0]**n)*(state[1]**m)
                    F.append(temp)
        F = np.asarray(F)
        F.shape = (len(F),1)
        return F           
    
    def evaluate_h(self,state):
        # produce polynomial features from normalized input state
        F = self.poly_features(state)

        # produce h function values
        h_eval = np.dot(self.W.T,F)
        return h_eval
    
    ### Q-learning function - version 1 - take random actions ###
    def train(self,**args):
        # make local (non-deep) copies of globals for easier reading
        grid = self.grid
        gamma = self.gamma
        
        # containers for storing various output
        self.training_episodes_history = {}
        self.training_reward = []
        self.validation_reward = []

        ### start main Q-learning loop ###
        for n in range(self.training_episodes): 
            # pick this episode's starting position
            grid.agent = self.training_start_schedule[n]
            
            ### get model functions evaluated at agent current state ###
            # get tuple location of agent and take poly transform
            h_eval = self.evaluate_h(np.copy(grid.agent))
                
            # update Q matrix while loc != goal
            episode_history = []      # container for storing this episode's journey
            total_episode_reward = 0
            for step in range(self.max_steps):   
                # update episode history container
                episode_history.append(grid.agent)
                
                ### if you reach the goal end current episode immediately
                if grid.agent == grid.goal:
                    break
                                   
                ### translate current agent location tuple into index
                s_k_1 = grid.state_tuple_to_index(grid.agent)
                    
                ### get action
                a_k = grid.get_action(method = self.action_method,h = h_eval,exploit_param = self.exploit_param)
                
                ### move based on this action
                s_k = grid.get_movin(action = a_k)
                
                ### update current location of agent 
                grid.agent = grid.state_index_to_tuple(state_index = s_k)
                
                ### get reward     
                r_k = grid.get_reward(state_index = s_k) 
                
                ### update model params ###
                # get poly features from new location
                h_eval = self.evaluate_h(np.copy(grid.agent))
                
                # update Q function data
                q_k = r_k + gamma*max(h_eval)
                
                # update model given new datapoint
                j = np.argmin(h_eval)
                h_j = h_eval[j]
                grad = self.poly_features(np.copy(grid.agent))    
                self.W[:,j] = self.W[:,j] - self.step_size*(h_j - q_k)*grad.flatten()    
                
                # update training reward
                total_episode_reward+=r_k
            # print out update if verbose set to True
            if 'verbose' in args:
                if args['verbose'] == True:
                    if np.mod(n+1,50) == 0:
                        print 'training episode ' + str(n+1) +  ' of ' + str(self.training_episodes) + ' complete'
            
            ### store this episode's training reward history
            self.training_episodes_history[str(n)] = episode_history
            self.training_reward.append(total_episode_reward)
            
            ### store this episode's validation reward history
            if 'validate' in args:
                if args['validate'] == True:
                    reward = self.validate(Q)
                    self.validation_reward.append(reward)
            
        print 'q-learning algorithm complete'
   
    ### run validation episodes ###
    def validate(self):
        # make local (non-deep) copies of globals for easier reading
        grid = self.grid
        
        # run validation episodes
        total_reward = []

        # run over validation episodes
        for i in range(self.validation_episodes):  

            # get this episode's starting position
            grid.agent = self.validation_start_schedule[i]

            # reward container for this episode
            episode_reward = 0

            # run over steps in single episode
            for j in range(grid.max_steps):
                ### if you reach the goal end current episode immediately
                if grid.agent == grid.goal:
                    break
                
                # evaluate functions at current location
                h_eval = self.evaluate_h(np.copy(grid.agent))

                # translate current agent location tuple into index
                s_k_1 = grid.state_tuple_to_index(grid.agent)
                    
                # get action
                a_k = grid.get_action(method = 'optimal',h = h_eval)
                
                # move based on this action - if move takes you out of gridworld don't move and instead move randomly 
                s_k = grid.get_movin(action = a_k, illegal_move_response = 'random')
  
                # compute reward and save
                r_k = grid.get_reward(state_index = s_k)          
                episode_reward += r_k
    
                # update agent location
                grid.agent = grid.state_index_to_tuple(state_index = s_k)
                
            # after each episode append to total reward
            total_reward.append(episode_reward)

        # return total reward
        return np.median(total_reward)