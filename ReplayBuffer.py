import torch;
import numpy as np;
import stable_baselines3
'''
Replay Buffer for GAE algorithm, uses
'''

#define the replay Buffer class
class RolloutBuffer():

    #constructor
    def __init__(self, max_memory:int, state_size:tuple, action_size:tuple, gamma:float = 0.99, beta:float = 100, gae_lambda:float=1):
        '''
        initializer of the rollout buffer to hold upto max_memory experience tuples with
        state_size and action_size shaped states and actions (both assumeed to be continuous) for the present policy.
        The experience tuples are discarded after the policy update.
        Rollout buffer is responsible storing state-action-logprob,reward predicted value function estimate and done flags,
        Rollout buffer responsible for generation batch of data from the memory using a given batchsize / return entire batch when no batch size is mentioned
        Rollout buffer responsible for computatation of GAE (schulman et al. 2015) and return from GAE using TD lambda
        '''

        #save variables
        self.max_memory = max_memory;
        self.state_size = state_size;
        self.action_size = action_size;
        self.memctr = 0;
        self.full =False; #boolean indicating the rolloub buffer is filled....
        self.gamma = gamma; #discount factor for the algorithm....in replay buffer for GAE calculation
        self.gae_lambda = gae_lambda; #lambda for TD lambda TD error calculation in GAE
        self.beta = beta;

        #define the replay memory as a fixed size numpy array.....for quick assignment and retrieval..
        self.state_memory = np.zeros(shape = (self.max_memory, *self.state_size));
        self.action_memory = np.zeros(shape = (self.max_memory, *self.action_size));
        self.logprob_memory = np.zeros(shape = (self.max_memory));
        self.reward_memory = np.zeros(shape = (self.max_memory));
        self.value_memory = np.zeros(shape =(self.max_memory)); #memory of value functions estimation...for GAE calculation
        self.done_memory = np.zeros(shape = (self.max_memory), dtype = bool); #present dones i.e the termination is after this state.....
        self.gae_advantages = np.zeros(shape = (self.max_memory));
        self.returns = np.zeros(shape = (self.max_memory));

        return;

    #replay buffer can load state, action, reward, next_state, termination tuples to memory
    def load_to_memory(self, state, action, logprob, reward, value, done):
        '''function loads the state, action -reward, next_state, and termination tuple to the memory buffer'''

        memindex = self.memctr # % self.max_memory;

        #directly go ahead and load the corresponding elements to the corresponding memory containers..at the latest index..
        self.state_memory[memindex] = state;
        self.action_memory[memindex] = action;
        self.logprob_memory[memindex] = logprob;
        self.reward_memory[memindex] = reward;
        self.value_memory[memindex] = value; #valu function estimate from the value function approximator....used for return calculation
        self.done_memory[memindex] = done; #present dones i.e the termination is after this state.....

        #now go ahead and increment memctr, modded with the memsize...to cycle over the memory
        self.memctr += 1;
        if(self.memctr == self.max_memory):
            self.full = True;
        self.memctr = self.memctr % self.max_memory;
        
        return;
    
    #define the function to reset the rollout_buffer
    def reset(self):
        self.state_memory = np.zeros(shape = (self.max_memory, *self.state_size));
        self.action_memory = np.zeros(shape = (self.max_memory, *self.action_size));
        self.logprob_memory = np.zeros(shape = (self.max_memory));
        self.reward_memory = np.zeros(shape = (self.max_memory));
        self.value_memory = np.zeros(shape =(self.max_memory)); #memory of value functions estimation...for GAE calculation
        self.done_memory = np.zeros(shape = (self.max_memory), dtype = bool); #present dones i.e the termination is after this state.....
        self.gae_advantages = np.zeros(shape = (self.max_memory));
        self.returns = np.zeros(shape = (self.max_memory));
        self.memctr = 0;
        self.full =False;


    #before anything else.....define a function fto generate the GAE estimate and returns forom the samples rollout...
    def gae(self, last_value:float):

        '''function computes the TD lambda estimate of the advantage according to the paper by Schulman : https://arxiv.org/pdf/1506.02438.pdf

        function computes n-step bootstrapped TD error and uses TD lambda for weighted advantage estimation.
        last_value: value function estimate corresponding to the next state at the last transition of the rollout (since it is not saved in the buffer)
        last_done : termination flag corresponding to the last_value's state...will be same as the last done element in rollout buffer
        '''

        

        #fisrtly initialize the gae_estimate for the last state in the rollout buffer...
        gae_adv = 0;
        #for the last state the TD error is the rb.r + gamma * last_value * (1-rb.termination[state])  - rb.value[state]
        for i in reversed(range(self.memctr, self.max_memory + self.memctr)):
            #get the step
            step = i % self.max_memory;
            if (step == self.memctr - 1):
                value_next = last_value;
            else:
                value_next = self.value_memory[(step + 1) % self.max_memory];
            non_terminal = 1 - int(self.done_memory[step]);
            #compute the TD error for the concerned state
            delta = self.reward_memory[step] + self.gamma*value_next*non_terminal + \
                (1/self.beta)*self.logprob_memory[step] - self.value_memory[step];
            #compund the gae_advantage
            gae_adv = delta + self.gamma*self.gae_lambda*non_terminal*gae_adv;
            self.gae_advantages[step] = gae_adv;
        
        #compute the returns
        self.returns = self.gae_advantages + self.value_memory;

        return;


    #replaybuffer can sample from buffer a set of experience tuples of size batch_size
    def get_data(self):
        '''
        function randomly permuted rollout_buffer data in a single batch...        
        '''

        assert(self.full), 'Buffer incompletely filled';
        
        #for sampling minibatches from the data....shuffle the buffer for random sampling
        inds = np.random.permutation(self.max_memory); #shuffles the indices 0 to self.max_memory - 1

        states = self.state_memory[inds];
        actions = self.action_memory[inds];
        logprobs = self.logprob_memory[inds];
        rewards = self.reward_memory[inds];
        values = self.value_memory[inds];
        dones = self.done_memory[inds];
        gae_advantages = self.gae_advantages[inds];
        returns = self.returns[inds];
        return [states, actions, logprobs, rewards, values, dones, gae_advantages, returns];
    
    #replaybuffer can sample from buffer a set of experience tuples of size batch_size
    def get_data_batched(self, batch_size:int = None):
        '''
        function randomly samples batch_size number of sampels from the replay buffer and returns
        the result as individual state, action, reward, next_state, done numpy tensors...        
        '''

        assert(self.full), 'Buffer incompletely filled';

        #check for the batch size....
        if(batch_size is None):
            #the entire memory is the batch
            batch_size = self.max_memory;
        
        #for sampling minibatches from the data....shuffle the buffer for random sampling
        inds = np.random.permutation(self.max_memory); #shuffles the indices 0 to self.max_memory - 1

        #noe go ahead and creat batch_size portions from the schuffled memory....in sequence
        start_ind = 0;
        while start_ind < self.max_memory:

            #create a batchsize potion of the memory;
            states = self.state_memory[inds[start_ind : start_ind+batch_size]];
            actions = self.action_memory[inds[start_ind : start_ind+batch_size]];
            logprobs = self.logprob_memory[inds[start_ind : start_ind+batch_size]];
            rewards = self.reward_memory[inds[start_ind : start_ind+batch_size]];
            values = self.value_memory[inds[start_ind : start_ind+batch_size]];
            dones = self.done_memory[inds[start_ind : start_ind+batch_size]];
            gae_advantages = self.gae_advantages[inds[start_ind : start_ind+batch_size]];
            returns = self.returns[inds[start_ind : start_ind+batch_size]];
            yield [states, actions, logprobs, rewards, values, dones, gae_advantages, returns];
            start_ind += batch_size;
        
        