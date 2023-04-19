from ast import Tuple
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spatial;
from gym import Env;
from gym import spaces
import scipy.io as scio;

#define the springMassSystem model
class SpringMassDamper(Env):

    #define the constructor
    def __init__(self, mass:float, springConst:float, dampingCoeff:float, ts:float) -> None:

        '''
        SpringMass Damper represents the simple discrete time spring mass damper system with mass spring and
        damping coefficient defined.ts represents the discretization of the time.
        '''

        super(SpringMassDamper, self).__init__();

        #save the variables
        self.mass = mass;
        self.springConst = springConst;
        self.dampingCoeff = dampingCoeff;
        self.ts = ts;
        self.step_cnt = 0;

        #define teh environment related variables here
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf,shape=(2,),dtype=np.float32);
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape=(2,),dtype=np.float32);
        self.reward_range = (-np.inf, np.inf);
        self.state = np.zeros(shape=(2,), dtype= np.float32);

    #the environment implements the reset function
    def reset(self):
        '''
        function performs a reset of the state to initialize the state of the spring between a
        box of +-1 along both axes
        '''

        self.state = np.random.uniform(low=-1,high=1, size=(2,));
        #reset the step count
        self.step_cnt = 0;
        
        return self.state.copy();

    #the environment implements the step function using an input action
    def step(self, action):
        
        '''
        function performs stepping using the ODE for the system using Range Kutta 4 from the latest state of the system
        '''

        #get the predicted state using hte action
        pred_state = action;

        #firstly get the slope at the present state.
        k11 = self.state[1];
        k12 = (-self.springConst/self.mass)*self.state[0] + (-self.dampingCoeff/self.mass)*self.state[1];

        #compute the 2nd set of points
        x2 = self.state[0] + (k11*self.ts/2);
        dx2 = self.state[1] + (k12*self.ts/2);

        #compute the 2nd slopes
        k21 = dx2;
        k22 = (-self.springConst/self.mass)*x2 + (-self.dampingCoeff/self.mass)*dx2;

        #compute the 3rd set of points
        x3 = self.state[0] + (k21*self.ts/2);
        dx3 = self.state[1] + (k22*self.ts/2);

        #compute the 3rd slopes
        k31 = dx3;
        k32 = (-self.springConst/self.mass)*x3 + (-self.dampingCoeff/self.mass)*dx3;

        #compute 4th set of points
        x4 = self.state[0] + (k31*self.ts);
        dx4 = self.state[1] + (k32*self.ts);

        #compute the 4th slopes
        k41 = dx4;
        k42 = (-self.springConst/self.mass)*x4 + (-self.dampingCoeff/self.mass)*dx4;

        #compute the next point estimates
        x = self.state[0] + (self.ts/6)*(k11+2*k21+2*k31+k41);
        dx = self.state[1] + (self.ts/6)*(k12+2*k22+2*k32+k42);
        self.state[0] = x;
        self.state[1] = dx;

        #the reward is negative of the predicteion error
        reward = -np.linalg.norm(self.state - pred_state, ord=2)**2;
        self.step_cnt+=1;
        if(self.step_cnt >= 500):
            done =True;
        else:
            done = False;
        
        return self.state, reward, done, {};
    
#define policy
def policy(state:np.ndarray, mass:float, springConst:float, dampingCoeff:float, ts:float):
    #firstly get the slope at the present state.
    state_out = state.copy();
    k11 = state[1];
    k12 = (-springConst/mass)*state[0] + (-dampingCoeff/mass)*state[1];

    #compute the 2nd set of points
    x2 = state[0] + (k11*ts/2);
    dx2 = state[1] + (k12*ts/2);

    #compute the 2nd slopes
    k21 = dx2;
    k22 = (-springConst/mass)*x2 + (-dampingCoeff/mass)*dx2;

    #compute the 3rd set of points
    x3 = state[0] + (k21*ts/2);
    dx3 = state[1] + (k22*ts/2);

    #compute the 3rd slopes
    k31 = dx3;
    k32 = (-springConst/mass)*x3 + (-dampingCoeff/mass)*dx3;

    #compute 4th set of points
    x4 = state[0] + (k31*ts);
    dx4 = state[1] + (k32*ts);

    #compute the 4th slopes
    k41 = dx4;
    k42 = (-springConst/mass)*x4 + (-dampingCoeff/mass)*dx4;

    #compute the next point estimates
    x = state[0] + (ts/6)*(k11+2*k21+2*k31+k41);
    dx = state[1] + (ts/6)*(k12+2*k22+2*k32+k42);
    state_out[0] = x;
    state_out[1] = dx;

    return state_out;


#main execution here,,
# if __name__ == '__main__':

#     #define an environment here
#     env = SpringMassDamper(1.0, 1.0, 0, 0.1);


#     init = env.reset();
#     y = init.copy();
#     score = 0;
#     done = False; i =0
#     while not done:
#         i+=1
#         #get action from the policy
#         action = policy(y, 1.0, 1.0, 0, 0.1);
#         y, r, done, info = env.step(np.zeros(shape=(2,), dtype = np.float32));
#         score += r;
#         print(f'step: {i:d}\t point: {y}\t\t reward : {r:.4f}');
#         init = np.vstack([init,y]);

#     print(f'score: {score:.4f}');
#     plt.plot(init[:,0]);
#     plt.plot(init[:,1]);
#     plt.show();
#     print('goodbye');

    