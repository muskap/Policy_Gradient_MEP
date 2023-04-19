import torch
import torch.nn as nn;
import torch.distributions as distr;
import numpy as np;
import os;
from gym import Env;
from utils import conjugate_gradient, linear_schedule, Visitation;
import pickle as pkl;

from Networks import Actor, Critic;
from ReplayBuffer import RolloutBuffer;


#define the main PPO Agent here...
'''
Deinition of Agent here is different from other models.
Here Agent maintains a handle to the environment to execute actions on the environment and collect rollout
Agent is responsibile for collecting a rollout of length rollout_length and storing the experience tuples in the rollout buffer.
Agent is then responsible for calling for computation of GAE on the rollout buffer using the collected rollout
Agent is then responsible for carrying out value and policy optimization with n_epochs number of gradient descent iterations
Agent is then responsible for carrying out this learning for a specified number of totaltimesteps
'''
class Agent():

    def __init__(
        self, state_size, action_size, max_action:float, max_memory:int, lr_critic:float= 3e-4, gamma:float=0.99,
        gae_lambda:float = 0.95, max_div=1e-3, n_epochs:int=10,
        max_ls_steps = 10, max_cg_iters=20, ls_decay_rate = 0.707, beta_schedule = linear_schedule(0, 245, 10, 1000)
    ) -> None:

        #save the variables...
        self.state_size = state_size;
        self.action_size = action_size;
        self.max_action = max_action;
        self.max_memory = max_memory;
        self.gamma = gamma;
        self.gae_lambda = gae_lambda;
        self.rollout_length = max_memory;
        self.n_epochs = n_epochs;
        self.max_div = max_div ; #constraint deviation upper bound
        self.max_ls_steps = max_ls_steps; #max_steps of line search for policy optimization
        self.max_cg_iters = max_cg_iters; #maximum number of conjugate gradient itereations for the conjugate gradient algoeithm
        self.ls_decay_rate = ls_decay_rate; #step decay rate for line search
        self.beta = 10.;
        self.beta_schedule = beta_schedule;
        
        self.state = None;
        self.score = 0;

        #define the actor critic and rolloutBUffers
        self.actor = Actor(self.state_size, self.action_size, max_action, name = 'actor');
        self.critic = Critic(self.state_size, lr = lr_critic, name='critic');
        self.rolloutBuffer = RolloutBuffer(self.max_memory, self.state_size, self.action_size, self.gamma, self.beta, self.gae_lambda);

        #for the agent...define a tensor to store the archive the old parameters of the actor...
        self.actor_params_archive = self.actor.flatten_params();
        #define a tensor to contain the distributions for the recent rollout...
        self.old_policy_mean = None;
        self.old_policy_logvar = None;

        self.training_logs = {'pol_entropy': [], 'probab_ratio': [], 'score': [],\
             'mus': [], 'logvars': [], 'kl_divs':[], 'sq_ratio_div':[], 'improvment':[], 'beta': [],
             'surr_loss_grad_norm':[], 'cg_final_residual_norm': []};
        self.visits:Visitation = None;

        pass;

    #agent can choose action for the state
    def choose_action(self, state:np.ndarray):

        state = torch.tensor(state.reshape(1,-1), dtype = torch.float).to(self.actor.device);

        #forward sample from the normal distribution
        action, logprob = self.actor.sample_normal(state, reparameterize = False);

        return action.cpu().numpy()[0], logprob.detach().cpu().numpy()[0];


    #agent can add experience tuples to memory
    def remember(self, state, action, logprob, reward, value, done):
        #just add the experience tuple to memory of the rollout_buffer
        self.rolloutBuffer.load_to_memory(state, action, logprob, reward, value, done);

    #define a function to compute the KL divergence between two distributions
    def kl_div(self, mean_a:torch.Tensor, logvar_a:torch.Tensor, mean_b:torch.Tensor, logvar_b:torch.Tensor) -> torch.Tensor:
        
        '''
        function computes the mean kl_divergence between two normal distributions a and b given defined by
        the mean and variances..the mean is computed accross the batch of samples....
        param mean_a: mean vector of the normal distribution a of dimension batch_size x n
        param logvar_a: standard deviation vector of normal distribution a of dimension batch_Size x n
        param mean_b: mean vector of the normal distribution b of dimension batch_size x n
        param logvar_b: standard deviation vector of normal distribution b of dimension batch_Size x n 
        '''

        #firstly define distributions using the arguments
        distr_a = distr.Normal(mean_a, logvar_a);
        distr_b = distr.Normal(mean_b, logvar_b);

        #compute the kl divergence..
        kl_div = distr.kl_divergence(distr_a, distr_b);
        kl_div = torch.mean(kl_div);
        log_logvar_a = torch.log(logvar_a);
        log_logvar_b = torch.log(logvar_b);

        #kl-divergence new to old
        kl = log_logvar_a - log_logvar_b + (logvar_b.pow(2) + (mean_b - mean_a).pow(2)) / (2.0 * logvar_a.pow(2)) - 0.5
        kl = kl.sum(1);
        kl = kl.mean();

        #kl-divergence old to new
        kl2 = log_logvar_b - log_logvar_a + (logvar_a.pow(2) + (mean_a - mean_b).pow(2)) / (2.0 * logvar_b.pow(2)) - 0.5
        kl2 = kl2.sum(1);
        kl2 = kl2.mean();

        return kl2;
    
    #define a function to compute the squared divergence of ration from 1
    def sq_ratio_div(self, logprob_a:torch.Tensor, logprob_b:torch.Tensor):

        '''
        function computes the mean squared divegence in the probability ratio between two distributions a and b
        from 1 i.e mean of (p_b/p_a -1)^2
        '''

        #from the probability ratio
        r_ba = torch.exp(logprob_b - logprob_a);
        #now compute the deviation of probability ratio from 1
        div = (r_ba - 1).pow(2);
        div = div.mean();

        return div;
    
    #define a function to compute the maximum squared divergence of ration from 1 using log sum exponent
    def max_sq_ratio_div(self, logprob_a:torch.Tensor, logprob_b:torch.Tensor):

        '''
        function computes the mean squared divegence in the probability ratio between two distributions a and b
        from 1 i.e mean of (p_b/p_a -1)^2
        '''

        #from the probability ratio
        r_ba = torch.exp(logprob_b - logprob_a);
        #now compute the deviation of probability ratio from 1
        div = (r_ba - 1).pow(2);
        div = torch.exp(100.0*div);
        div = torch.log(div.sum())/100.0;

        return div;

    #define a function to compute the fissher vector product to compute divergence hessian w.r.t times a vector
    def fvp(self, states:torch.Tensor, v:torch.Tensor) -> torch.Tensor:

        '''
        function computes the matrix product of the kl-divergence hessain w.r.t the actor's parameters
        with a vector....
        '''

        damping =1e-2;

        #befire computing the kl-divergence between the reference and the present policy...obtain
        # the policy distributions for the states in the rollout..
        policy_mean, policy_logvar = self.actor.forward(states);
        #firstly get the kl-divergence between the distributions
        mean_kl_div = self.kl_div(self.old_policy_mean, self.old_policy_logvar, policy_mean, policy_logvar);

        #now before computing the fvp...get the parameters w.r.t which the gradient is to be obtained..
        #i.e actor's parameters..flattened..
        # flat_params = self.actor.flatten_params();

        #now get the gradient of kl-divergence w.r.t flat_params
        grad = torch.autograd.grad(mean_kl_div, self.actor.parameters(), create_graph=True);
        grad = torch.cat([g.view(-1) for g in grad]);

        #now multiply gradient and the vector what we have in the argument..elementwise
        grad = torch.dot(grad , v);
        
        #the result of Hx is the gradient of the grad w.r.t parameters..
        grad2 = torch.autograd.grad(grad, self.actor.parameters(), create_graph = False);
        grad2 = torch.cat([g.contiguous().view(-1) for g in grad2]);

        return grad2 + damping*v;
    
    #define a function to compute the fissher vector product to compute sq_ratio_divergence hessian w.r.t times a vector
    def fvp_sq_ratio(self, states:torch.Tensor, actions:torch.Tensor, logprobs:torch.Tensor, v:torch.Tensor) -> torch.Tensor:

        '''
        function computes the matrix product of the sq-ratio-divergence hessain w.r.t the actor's parameters
        with a vector....
        '''

        damping =1e-2;

        #befire computing the kl-divergence between the reference and the present policy...obtain
        # the policy distributions for the states in the rollout..
        policy_mean, policy_logvar = self.actor.forward(states);
        dist = distr.Normal(policy_mean, policy_logvar);
        log_probs = dist.log_prob(actions); log_probs = log_probs.sum(1);

        #firstly get the pr-divergence between the distributions
        # mean_kl_div = self.kl_div(self.old_policy_mean, self.old_policy_logvar, policy_mean, policy_logvar);
        mean_pr_div = self.sq_ratio_div(logprobs, log_probs);

        #now before computing the fvp...get the parameters w.r.t which the gradient is to be obtained..
        #i.e actor's parameters..flattened..
        # flat_params = self.actor.flatten_params();

        #now get the gradient of kl-divergence w.r.t flat_params
        grad = torch.autograd.grad(mean_pr_div, self.actor.parameters(), create_graph=True);
        grad = torch.cat([g.view(-1) for g in grad]);

        #now multiply gradient and the vector what we have in the argument..elementwise
        grad = torch.dot(grad , v);
        
        #the result of Hx is the gradient of the grad w.r.t parameters..
        grad2 = torch.autograd.grad(grad, self.actor.parameters(), create_graph = False);
        grad2 = torch.cat([g.contiguous().view(-1) for g in grad2]);

        return grad2 + damping*v;
    
    #define a function to compute the fissher vector product to compute max_sq_ratio_divergence hessian w.r.t times a vector
    def fvp_max_sq_ratio(self, states:torch.Tensor, actions:torch.Tensor, logprobs:torch.Tensor, v:torch.Tensor) -> torch.Tensor:

        '''
        function computes the matrix product of the sq-ratio-divergence hessain w.r.t the actor's parameters
        with a vector....
        '''

        damping =1e-2;

        #befire computing the kl-divergence between the reference and the present policy...obtain
        # the policy distributions for the states in the rollout..
        policy_mean, policy_logvar = self.actor.forward(states);
        dist = distr.Normal(policy_mean, policy_logvar);
        log_probs = dist.log_prob(actions); log_probs = log_probs.sum(1);

        #firstly get the pr-divergence between the distributions
        # mean_kl_div = self.kl_div(self.old_policy_mean, self.old_policy_logvar, policy_mean, policy_logvar);
        mean_pr_div = self.max_sq_ratio_div(logprobs, log_probs);

        #now before computing the fvp...get the parameters w.r.t which the gradient is to be obtained..
        #i.e actor's parameters..flattened..
        # flat_params = self.actor.flatten_params();

        #now get the gradient of kl-divergence w.r.t flat_params
        grad = torch.autograd.grad(mean_pr_div, self.actor.parameters(), create_graph=True);
        grad = torch.cat([g.view(-1) for g in grad]);

        #now multiply gradient and the vector what we have in the argument..elementwise
        grad = torch.dot(grad , v);
        
        #the result of Hx is the gradient of the grad w.r.t parameters..
        grad2 = torch.autograd.grad(grad, self.actor.parameters(), create_graph = False);
        grad2 = torch.cat([g.contiguous().view(-1) for g in grad2]);

        return grad2 + damping*v;
    
    #define a function to compute the surrogate loss for the state and emperiical advantage
    def actorloss(self, states:torch.Tensor, actions:torch.Tensor, logprobs:torch.Tensor, gae_advantages:torch.Tensor):

        '''
        function computes the importance sampled surrogate loss function for the state-actoin trajectory for
        the actor's policy...
        param states: states visted in the rollout...
        param actions: actions according to the policy used for collecting the rollout trajectory
        param logprobs: log probabilities of the actions for the states for the policy ued for collecting the rollout
        param gae_advantages: emperical advantages for the collected rollout 
        '''

        #compute the logprob of the sampled actions under the new policy
        mus, logvars = self.actor.forward(states);
        dist = distr.Normal(mus, logvars);
        log_probs = dist.log_prob(actions);
        log_probs = log_probs.sum(1); #total log_probability...includes gradient information
        entropy = list(dist.entropy().detach().cpu().numpy().flatten()); #for logging purposes...

        #so coming to actor optimizarion...
        ratio = torch.exp(log_probs - logprobs);
        #now normalize the gae_advantages for reduced gradients...
        gae_advantages = gae_advantages - gae_advantages.mean() / (gae_advantages.std() + 1e-8);
        actor_loss1 = ratio * gae_advantages;

        log_logvar_a = torch.log(self.old_policy_logvar);
        log_logvar_b = torch.log(logvars);

        #kl-divergence new to old
        kl = log_logvar_a - log_logvar_b + (logvars.pow(2) + (mus - self.old_policy_mean).pow(2)) / (2.0 * self.old_policy_logvar.pow(2)) - 0.5
        kl = kl.sum(1);
        # kl = kl.mean();

        actor_loss1 = torch.mean(actor_loss1 + (1/self.beta)*kl);

        return actor_loss1;
    
    #define a function to collect_rollouts from the environment using the policy...for
    def collect_rollout(self, env:Env, n_rollout_steps:int=None):
        '''
        function collects a rollout trajectory for n_rollout_steps on the environment using the agents policy
        '''

        if n_rollout_steps is None:
            n_rollout_steps = self.rollout_length;
        
        # firstly reset the rolloutBuffer and check whether the present state is maintained in the agent
        assert self.state is not None, 'Present state reference unavaiable for rollout execution';

        #reset the rollout_buffer
        n_steps = 0;
        self.rolloutBuffer.reset();

        #now carry out the rollout
        while n_steps < n_rollout_steps:

            # fisrtly choose the action for the present state
            action, logprob = self.choose_action(self.state);

            #execute the action on the environment
            state_, reward, done, info = env.step(action);
            reward *= -1;
            self.score += reward;

            #now before adding to rollout buffer, compute the estimates value function from the present state
            # and removing gradient information
            with torch.no_grad():
                value = self.critic.forward(torch.tensor(self.state.reshape(1,-1), dtype=torch.float).to(self.critic.device)).view(-1)
                value = value.cpu().numpy()[0];

            #now add it to the rolloutbuffer
            self.remember(self.state, action, logprob, reward, value, done);
            #add to the visitation log
            if(self.visits is not None):
                self.visits.log_state(self.state);


            # print(f'step: %i'%n_steps, ',state: {self.state}, action: %.2f ' %action, ', reward: %.2f' %reward, ', done: %i' %done);

            #if termination is reached then reset the environment and set that as the state
            if(done):
                self.state = env.reset();
                print(f'score: {self.score}');
                self.training_logs['score'].append(self.score);
                self.score = 0;
            else:
                self.state = state_;

            n_steps +=1;

        #before returning, compute the gae advantages and returns from the rollout
        #with the last state, compute the value function estimate for the last state and use it for computation of GAE estimates and returns 
        with torch.no_grad():
            last_value = self.critic.forward(torch.tensor(self.state.reshape(1,-1), dtype=torch.float).to(self.critic.device)).view(-1);
            last_value = last_value.cpu().numpy()[0];
        #compute the gae estimate using this
        self.rolloutBuffer.gae(last_value);
        
        #finally return the number of timesteps executed
        return n_steps;
    
    #now the agent can train the actor and critic.using the clipped policy gradient theorem
    def train(self, n_epochs:int = None):
        '''
        function implements n_epochs of training of actor and critic using the PPO algorithm...
        '''

        print('-'*25);

        #firstly check for the presence of the epochs argument
        if(n_epochs is None):
            n_epochs = self.n_epochs;
        
        #for training using TRPO, we first start with loading the rollout trajectory..
        states, actions, logprobs, rewards, values, dones, gae_advantages, returns = self.rolloutBuffer.get_data();
        #convert rollout trajectory data into tensors
        states = torch.tensor(states, dtype=torch.float).to(self.actor.device);
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device);
        logprobs = torch.tensor(logprobs, dtype = torch.float).to(self.actor.device);
        rewards = torch.tensor(rewards, dtype = torch.float).to(self.actor.device);
        values = torch.tensor(values, dtype = torch.float).to(self.actor.device);
        dones = torch.tensor(dones, dtype=torch.bool).to(self.actor.device);
        gae_advantages = torch.tensor(gae_advantages, dtype= torch.float).to(self.actor.device);
        returns = torch.tensor(returns, dtype = torch.float).to(self.actor.device);

        #firstly set the old_policy mean and variance according for the policy that generated the rollout
        with torch.no_grad():
            self.old_policy_mean, self.old_policy_logvar = self.actor.forward(states);

        #firstly to optimize the actor.....compute the actor loss using the gae_advnatage and
        # compute gradient w.r.t parameters..
        #compute actor surrogate loss
        loss = self.actorloss(states, actions, logprobs, gae_advantages);
        #now before computing the gradient w.r.t actor parameters...obtain the parameters as a flattened vector
        flat_params_actor = self.actor.flatten_params();
        #now compute the gradient w.r.t actor's parameters.....this becomes the rhs for conjugate gradient
        g = torch.autograd.grad(loss, self.actor.parameters());
        g = torch.cat([g_t.view(-1) for g_t in g]);
        with torch.no_grad():
            self.training_logs['surr_loss_grad_norm'].append(torch.norm(g).item());

        #with the gradient of the surrogate loss w.r.t parameters found...time to compute the step direction
        #using the conjugate gradient...before that define a lambda function for the fvp with the states asssigned
        # fvpl = lambda x : self.fvp(states, x);
        fvpl = lambda x : self.fvp_sq_ratio(states, actions, logprobs, x);
        # fvpl = lambda x : self.fvp_max_sq_ratio(states, actions, logprobs, x);
        #finally call the conjugate gradient algorithm to solve for the step direction
        step_dir, rdr = conjugate_gradient(fvpl, -g, max_iters=self.max_cg_iters);
        with torch.no_grad():
            self.training_logs['cg_final_residual_norm'].append(rdr.item());
        print('-'*10 +'conjugate gradient complete'+'-'*10);
        self.f.write('-'*10 +'conjugate gradient complete'+'-'*10+'\r\n');
        #now determine the amount of movement of parameters in the direction of the step_dir
        shs =  torch.dot(step_dir, fvpl(step_dir));
        step_amt = step_dir * torch.sqrt(2*self.max_div / abs(shs)); #the amount of stepping for full step...

        #finally begin linesearch for determination of actual improvement along the step and
        # decay step if required
        #start with a step size of 1
        step_size = 1.0;
        expected_improvement = torch.dot(g, step_amt);
        #now loop for max_steps or until sufficiency..
        for st in range(self.max_ls_steps):

            #for each step...compute the proposed parameter along the step direction by step_size fraction
            prosp_flat_params_actor = flat_params_actor + step_size*step_amt;

            #now check for surrogate loss and kl-divergence constraint...
            #for that first set the prospective parameters to the actor...
            self.actor.set_flat_params(prosp_flat_params_actor);

            #using this compute the actor loss i.e surrogate loss and the kl-divergence...
            new_loss = self.actorloss(states, actions, logprobs, gae_advantages);
            improvement = new_loss - loss;
            #to compute the kl-divergence....get the probability distributions
            prosp_policy_mean, prosp_policy_logvar = self.actor.forward(states);
            #now the kl_divergence
            kl_div = self.kl_div(self.old_policy_mean, self.old_policy_logvar, prosp_policy_mean, prosp_policy_logvar);
            #also compute the probability ratio divergence for reference
            prosp_dist = distr.Normal(prosp_policy_mean, prosp_policy_logvar);
            prosp_log_probs = prosp_dist.log_prob(actions); prosp_log_probs = prosp_log_probs.sum(1);
            pr_div = self.sq_ratio_div(logprobs, prosp_log_probs);
            max_pr_div = self.max_sq_ratio_div(logprobs, prosp_log_probs);

            print(f'improvement | Expected: {expected_improvement.item():.4f}\tActual:{improvement.item():.4f} \
                | mean_kl_div: {kl_div.item():.4f}, ratio_div: {torch.sqrt(pr_div).item():.4f}, max_ratio_div: {torch.sqrt(max_pr_div).item():.4f}');
            self.f.write(f'improvement | Expected: {expected_improvement.item():.4f}\tActual:{improvement.item():.4f} \
                | mean_kl_div: {kl_div.item():.4f}, ratio_div: {torch.sqrt(pr_div).item():.4f}, max_ratio_div: {torch.sqrt(max_pr_div).item():.4f}\r\n');    

            #now the checks
            if(not np.isfinite(new_loss.item())):
                print(f'Houston we have a problem! infinite loss found');
                self.f.write(f'Houston we have a problem! infinite loss found\r\n');
            elif(improvement > 0):
                print(f'Not improvement found..shrinking step');
                self.f.write(f'Not improvement found..shrinking step\r\n');
            elif(pr_div > self.max_div*1.1):
                print(f'kl divergence constraint voilated ... shrinking step');
                self.f.write(f'kl divergence constraint voilated ... shrinking step\r\n');
            else:
                print('looks okay');
                self.f.write('looks okay\r\n');
                self.training_logs['kl_divs'].append(kl_div.item());
                self.training_logs['sq_ratio_div'].append(pr_div.item());
                self.training_logs['improvment'].append(improvement.item());
                self.training_logs['beta'].append(self.beta);
                break;
            step_size *= self.ls_decay_rate;
        else:
            #step improvement failed to get improvement or contraint satisfaction...revert changes
            print('step improvement failed to get improvement or contraint satisfaction...reverting changes')
            self.f.write('step improvement failed to get improvement or contraint satisfaction...reverting changes\r\n')
            self.actor.set_flat_params(flat_params_actor);
            self.training_logs['kl_divs'].append(kl_div.item());
            self.training_logs['sq_ratio_div'].append(pr_div.item());
            self.training_logs['improvment'].append(improvement.item());
            self.training_logs['beta'].append(self.beta);
        print('-'*10 + 'line-search complete' + '-'*10);
        self.f.write('-'*10 + 'line-search complete' + '-'*10 + '\r\n');

        
        #now carry out n epochs of training
        for epoch in range(n_epochs):

            #for each epoch....do a complete pass over the minibatches of data..
            for states, actions, logprobs, rewards, values, dones, gae_advantages, returns \
                in self.rolloutBuffer.get_data_batched(batch_size=128):

                #we are in a minibatch of data...now compute the required losses....

                #convert rollout trajectory data into tensors
                states = torch.tensor(states, dtype=torch.float).to(self.actor.device);
                actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device);
                logprobs = torch.tensor(logprobs, dtype = torch.float).to(self.actor.device);
                rewards = torch.tensor(rewards, dtype = torch.float).to(self.actor.device);
                values = torch.tensor(values, dtype = torch.float).to(self.actor.device);
                dones = torch.tensor(dones, dtype=torch.bool).to(self.actor.device);
                gae_advantages = torch.tensor(gae_advantages, dtype= torch.float).to(self.actor.device);
                returns = torch.tensor(returns, dtype = torch.float).to(self.actor.device);

                #firstly compute the predictions.....be sure to maintain gradient information
                #predict the value function
                values_pred = self.critic.forward(states).view(-1);

                #now optimizing the critic...
                critic_loss = nn.functional.mse_loss(values_pred, returns);
                self.critic.optimizer.zero_grad();
                critic_loss.backward(retain_graph = True);
                self.critic.optimizer.step();

                #done with optimization in the minibatch step...
                # store values for loggin purposes..
            

    #define a function for the agent to learn for a given number of timesteps, using a given rollout length
    def learn(self, env:Env, n_learn_steps:int, n_rollout_steps:int=None):
        
        '''
        function implements learning from the envionment env for total number of timesteps given by n_learn_steps,
        rollouts of n_rollout_steps are collected for each rollout run rolout runs keep on executing until the total number of steps
        exceed the required total number of steps for learning...
        '''

        #check if the rollout length is passed in as an argument
        if n_rollout_steps is None:
            #take agent's rollout_length as the default value
            n_rollout_steps = self.rollout_length;
        
        #start with total number of steps as zero
        n_steps = 0;
        i=0;

        self.f = open('tmp/trpo/logs.txt', 'w');
        
        while n_steps < n_learn_steps:

            #set beta according to the schedule
            self.beta = self.beta_schedule(i);
            self.rolloutBuffer.beta = self.beta;

            steps = self.collect_rollout(env, n_rollout_steps=n_rollout_steps);

            self.train(n_epochs=10);

            print(f'rollout run: %i' %(i+1), 'Compleed');
            self.f.write(f'rollout run: {(i+1)} Compleed\r\n');

            #increment the n_steps and the rollout run count
            n_steps += steps;
            i+=1;
        
        self.f.close();
        
        return;

    #agent can save the models
    def save_models(self):
        print('.... saving models ....');
        self.actor.save_checkpoint();
        self.critic.save_checkpoint();
        f = open('tmp/trpo/agent_training_logs.pkg', 'wb');
        pkl.dump(self.training_logs, f);
        f.close();

        
    def load_models(self):
        print('.... loading models ....');
        self.actor.load_checkpoint();
        self.critic.load_checkpoint();
        # f = open('tmp/trpo/agent_training_logs.pkg', 'rb');
        # # self.training_logs = pkl.load(f);
        # f.close();

    
