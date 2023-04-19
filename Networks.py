import torch;
import torch.nn as nn;
import torch.distributions as distr;
import os;
import numpy as np;

#networks defined here..
class Actor(nn.Module):

    '''
    Actor implements the paramertrized policy as a function approximator represented using a deep network.
    it takes in the state (state dimensional vector i.e 1d tensor) as input and returns the parameters of
    a continouus probability distribution accross the action space (action dimensional probability distribution).
    present Actor class implements a gaussian distribution accross the action space, returning the mean and variance
    action_dimensional vectors for the state imput
    '''

    def __init__(self, state_size:tuple, action_size:tuple, max_action:float, fc1_dims =64, fc2_dims = 64, name='actor', chkpt_dir = 'tmp\\trpo') -> None:
        super(Actor, self).__init__();

        #save the variables
        self.state_size = state_size;
        self.action_size = action_size;
        self.max_action = max_action;
        self.fc1_dims = fc1_dims;
        self.fc2_dims = fc2_dims;
        self.name = name;
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_trpo');
        self.reparam_noise = 1e-6;

        #define the network...
        self.lin1 = nn.Linear(*self.state_size, self.fc1_dims);
        self.lin2 = nn.Linear(self.fc1_dims, self.fc2_dims);
        self.mu = nn.Linear(self.fc2_dims, *self.action_size);
        # self.logvar = nn.Linear(self.fc2_dims, *self.action_size);
        self.logvar = nn.Parameter(torch.ones(self.action_size, dtype=torch.float)*0, requires_grad=True); #defining a logvariace parameters

        #define the optimizer and device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu';
        self.to(self.device);

        return;
    
    #define the forward propogation function
    def forward(self, state):
        '''
        function implements forward propogation on the policy network, to get the mean and logvariance for the state input
        '''

        out1 = self.lin1(state);
        out1r = torch.tanh(out1);
        out2 = self.lin2(out1r);
        out2r = torch.tanh(out2);

        mu = self.mu(out2r);
        mu = torch.tanh(mu)*self.max_action;
        # logvar = self.logvar(out2r);
        # logvar = torch.sigmoid(logvar)*self.max_action;
        # logvar  = torch.ones_like(mu)*0.2; #testing with a constant variance....
        logvar = torch.ones_like(mu) * torch.exp(self.logvar); #definiting logvar as a individual parameter to be optimized

        return mu, logvar;

    
    #define a function to sample from the policy
    def sample_normal(self, state, reparameterize=True):
        '''
        function samples from the policy after forward propogation through the state
        reparamterize is the argument to allow gradient computation through sampling using
        reperametrization trick
        '''

        #firstly, forward propogate through the network to get the mean and logvar of the policy
        mu, logvar = self.forward(state);

        #now sample from the normal distribution defined by the mean and standard deviation
        # logvar = torch.exp(logvar);
        normaldist = distr.Normal(mu, logvar);
        if reparameterize == True:
            actions = normaldist.rsample();#action_size dimensional action sample
        else:
            actions = normaldist.sample();
            
        actions.clamp_(-self.max_action, self.max_action)
        #rescale the actions to [-1,1] and upscale to max_actions range.....done possibly for normalization of model for difference action ranges
        logprobs = normaldist.log_prob(actions);
        # actions = torch.tanh(actions);
        # logprobs -= torch.log(1-actions.pow(2)+self.reparam_noise)
        logprobs = logprobs.sum(1, keepdim=True); #total log-probability of the action...
        # actions = actions*torch.tensor(self.max_action).to(self.device)

        return actions, logprobs;
    
    #define a function to flatten the model parameters..
    def flatten_params(self) -> torch.Tensor:

        '''
        function flattens the model parameters into a 1-d vector for computatation of gradient of losses and hessians
        '''

        #just loop through the parameters and get the parameters values
        flat_params = torch.cat([p.view(-1) for p in self.parameters()]);

        return flat_params;
    
    #set flattend params to model
    def set_flat_params(self, flat_params:torch.Tensor):

        '''
        function sets the parameters values passed as arguments to the model parameters, the original parameters with replacement....
        '''

        #directly loop through the parameters and set them to the model parameters...
        with torch.no_grad():
            totalLength = 0;
            for p in self.parameters():
                #get the parameter length
                sz = p.size();
                len = np.prod(sz);
                #set the parameters from totalLength to totalLEngth+len to the parameter after scaling
                p.copy_(flat_params[totalLength:totalLength+len].view(sz));
                totalLength += len;
    
    #finally some helper functions to save the actor model at checkpoints
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file);

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file));


#define the critic now...
class Critic(nn.Module):
    '''
    Critic represents the value function approximation implemented using a deep network.
    it takes in state-dimensional vector as input and returns a scalar output for the value function approximation
    '''

    #constructor
    def __init__(self, state_size:tuple, lr:float, fc1_dims=64, fc2_dims=64, name = 'critic', chkpt_dir = 'tmp\\trpo') -> None:
        super(Critic, self).__init__();

        #save the variables
        self.state_size = state_size;
        self.fc1_dims = fc1_dims;
        self.fc2_dims = fc2_dims;
        self.name = name;
        self.checkpoint_file = os.path.join(chkpt_dir, self.name+'_trpo');

        #define the network
        self.lin1 = nn.Linear(*self.state_size, self.fc1_dims);
        self.lin2 = nn.Linear(self.fc1_dims, self.fc2_dims);
        self.lin3 = nn.Linear(self.fc2_dims, 1);

        #define the optimizer and device
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr);
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu';
        self.to(self.device);

        return;
    
    #define the forward propogation function
    def forward(self, state):
        '''
        function forward propogates through the critic to compute the value function estimate for the state
        '''

        out1 = self.lin1(state);
        out1r = torch.relu(out1);
        out2 = self.lin2(out1r);
        out2r = torch.relu(out2);
        value = self.lin3(out2r);

        return value;
    
    #define a function to flatten the model parameters..
    def flatten_params(self) -> torch.Tensor:

        '''
        function flattens the model parameters into a 1-d vector for computatation of gradient of losses and hessians
        '''

        #just loop through the parameters and get the parameters values
        flat_params = torch.cat([p.view(-1) for p in self.parameters()]);

        return flat_params;
    
    #set flattend params to model
    def set_flat_params(self, flat_params:torch.Tensor):

        '''
        function sets the parameters values passed as arguments to the model parameters, the original parameters with replacement....
        '''

        #directly loop through the parameters and set them to the model parameters...
        with torch.no_grad():
            totalLength = 0;
            for p in self.parameters():
                #get the parameter length
                sz = p.size();
                len = np.prod(sz);
                #set the parameters from totalLength to totalLEngth+len to the parameter after scaling
                p.copy_(flat_params[totalLength:totalLength+len].view(sz));
                totalLength += len;
    
    #define the helper functions...
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file);

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file));


