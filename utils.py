import numpy as np;
import torch;
import torch.nn as nn;
import torch.distributions as distr;
import matplotlib.pyplot as plt;


#now define the conjugate gradient algoeithm
def conjugate_gradient(Ax:callable, b:torch.Tensor, x0:torch.Tensor= None, max_iters:int=10, error_thresh:float = 1e-6):

    '''
    function performs conjugate gradient algoeithm to find the solution of the linear equation Ax = b.
    param Ax: is a callable function that returns the result of Ax flattened into a 1d vector...
    param b: is the rhs constant 1d vector of size (n,)
    param x0: initial assumption of the solution (Optional). must be of same size as b 
    '''

    # f = open('tmp/trpo/.txt', 'a');

    #firstly start with an initial assumption if none is provided
    if(x0 is None):
        x = torch.zeros_like(b);
    else:
        assert (x0.shape == b.shape),'Sizes of b and x0 do not match';
        x = x0.clone();
    #initialize the residual and projection
    r = (b - Ax(x));
    p = r.clone();
    rdr = torch.dot(r,r);
    for iter in range(max_iters):
        
        Ap = Ax(p);
        alpha = rdr / torch.dot(p, Ap);

        x += alpha * p;
        r -= alpha * Ap;

        rdr_ = torch.dot(r, r);
        print(f'iter: {iter} residual norm:{rdr_:.6f}');
        
        #check for r^Tr..
        if(rdr_ < error_thresh):
            break;
        
        #otherwise continue...
        beta = rdr_/rdr;
        p = r + beta * p;
        rdr = rdr_;
        
    return x, rdr;


#define a function to carry out linear search along the direction of the step
def linesearch(constraint:callable, objective:callable, step_amt:torch.Tensor, init_param:torch.Tensor, constr_lim:float, max_steps:int=10,\
    decayrate:float = 0.707):

    '''
    function performns linear search using the objective function and quadratic constraint function to determine
    the optimzal solutino that satisfies the constraint bound (uppoer bound)
    param constraint: constraint function for checking the upper bound voilation
    param objective: objective function to be minimized....used to check for improvement
    param step_amt: full step amount in the paarmeters
    param init_param: starting parameter value befire stepping..
    param constr_lim: constraint upper bound
    param max_steps: maximum number of decay linearsearch steps
    param decayrate: rate of decay for the stepsize in linesearch
    '''

    step_size = 1.0;
    init_obj_fval = objective(init_param);

    for iter in range(max_steps):

        param = init_param + step_size*step_amt;

        #now check the objective function and constraint saticfaction...
        func_val = objective(param);
        constr = constraint(param);
        improvement = func_val - init_obj_fval;

        #check for improvement and constraint...
        if (improvement > 0):
            #no improvement...
            print('improvement not made, reducing step size');
            step_size *= decayrate;
        elif (constr > constr_lim*1.01):
            #constraint not satisfied...
            print('constraint voilated, reducing step size');
            step_size *= decayrate;
        else:
            print('works...');
            return param, step_size;
    #if nothing works...return the original param as the result..
    return init_param, 0.;


#define a linear schedule
def linear_schedule(w_start:float, w_end:float, h_min:float, h_max:float):
    '''
    function implements a liner schedule starting from h_offset going from h_min to h_max starting at w_start and ending at w_end
    '''

    return lambda x : np.clip(h_min + (h_max-h_min)*(x - w_start)/(w_end - w_start), h_min, h_max);


#define a class for holding the state visitation log for the  environment
class Visitation():

    #define the constructor
    def __init__(self, min_max_tuple:tuple, n_bins:int =10000) -> None:

        '''
        Visitation is a class that hold an n-dimensional state visitation log for the environment under action by an agent..
        param min_max_tuple: tuple for min and max along each dimension. for eg: ((-1,-2.3), (1, 3.5)) for
        -1,1 min max along 1st dimension and -2.5 3.5 along the 2nd dimension. 
        param n_bins: number of discrete binis along each dimension....uniform for now...
        '''

        self.min_max_tuple = min_max_tuple;
        self.n_dims = len(self.min_max_tuple[0]);
        self.n_bins =  n_bins;

        #define a n-dimensional array to hold the visitations
        self.visits = np.zeros(shape = (int(self.n_bins)+1,)*self.n_dims);

        return;
    
    #define a function to insert a state into the 
    def log_state(self, state:np.ndarray):

        #get the bin placement of the state along each dimension of the visitation array.
        placement = np.zeros(shape= (self.n_dims,), dtype=int);

        for i in range(len(self.min_max_tuple)):

            #get the placement along the ith dimension...
            assert((state[i] >= self.min_max_tuple[0][i]) & (state[i] <= self.min_max_tuple[1][i])), 'out of Bounds';

            placement[i] = int( self.n_bins*(state[i] - self.min_max_tuple[0][1])/(self.min_max_tuple[1][i] - self.min_max_tuple[0][i]) );
        
        self.visits[tuple(placement)] += 1;

        return;


#define a function to make a confidence interval plot
def plotConfInterval(xData:np.ndarray, meanData: np.ndarray, stdData:np.ndarray, axes:plt.Axes, color:str=None, label:str=None):

    under_line     = (meanData-stdData)
    over_line      = (meanData+stdData)

    #Plotting:
    axes.plot(xData, meanData, linewidth=1.5, color = color, label = label) #mean curve.
    axes.fill_between(xData, under_line, over_line, color=color, alpha=.1) #std curves.
    axes.legend();

if __name__ == '__main__':

    #define a dummy model...
    # visitation_test = Visitation(((0, 1), (1, 2)), n_bins = 100);

    # for i in range(int(1e5)):

    #     #generate random number..
    #     num = np.random.randn(2)*0.1 + [0.5,1.5];
    #     print(f'number : {num}');

    #     visitation_test.log_state(num);
    
    # print(visitation_test.visits);
    # plt.contourf(visitation_test.visits);
    # plt.show();

    #testing the confidence interval plot..
    f = plt.figure(figsize=(16,9));
    ax = f.subplots(nrows =1, ncols=1);
    mean = np.arange(0, 100, 0.1);
    std = np.ones_like(mean);
    plotConfInterval(mean, std, ax, color='b');

    print('goodbye tata');