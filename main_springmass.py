import pybullet_envs;
import gym;
from PPO_Agent import Agent;
from springMassTest import SpringMassDamper;

import numpy as np;
import os;
import matplotlib.pyplot as plt;


#define the main function here
if __name__ == "__main__":

    # env = gym.make('InvertedPendulumBulletEnv-v0');
    # env = gym.make('Pendulum-v1');
    env = SpringMassDamper(mass=1.0, springConst=1.0, dampingCoeff=0, ts = 0.1);
    agent = Agent(env.observation_space.shape, env.action_space.shape, env.action_space.high[0], 2048, max_div=1e-2);

    agent.state = env.reset();

    # env.render(mode = 'human');

    #before starting...create a directory for saving the agent..
    name = 'springmass';
    trial_number = '0';
    filename = os.path.join('tmp','trpo_'+ name+'_trial'+trial_number);

    try:
        os.mkdir('tmp/trpo');
    except:
        print('tmp/ppo Already Exists. Overwriting the directory');


    #learn for 512000 steps i.e 250 rollout runs
    agent.learn(env, n_learn_steps=512000, n_rollout_steps=2048); # 307200
    
    plt.plot(agent.training_logs['score']);
    agent.save_models();

    os.rename('tmp\\trpo', filename);

    #run the model
    agent.load_models();

    #run for predefined number of episodes
    f = plt.figure(); axes = f.subplots(1); ax = plt.plot(np.arange(501), np.zeros(501), '-r', np.arange(501), np.zeros(501), '-b',\
        np.arange(1,501), np.zeros(500), '-k',\
        np.arange(1,501), np.zeros(500), '-m');
    axes.legend(labels= ['pos', 'vel', 'a_pos', 'a_vel']);
    axes.set_ylim(-1.5, 1.5);
    for epi in range(100):
        observation = env.reset();#env.render();
        pts_x = [observation[0]]; pts_y = [observation[0]]; pts_ax = []; pts_ay=[];
        done = False;
        score = 0;
        while not done:
            action, logprob = agent.choose_action(observation);
            observation_, reward, done, info = env.step(action);
            pts_x.append(observation_[0]); pts_y.append(observation_[1]);
            pts_ax.append(action[0]); pts_ay.append(action[1]);
            #env.render();
            score+=reward;

            observation = observation_;
        print(f'episode: {epi}, score: {score:.4f}');
        ax[0].set_ydata(pts_x);
        ax[1].set_ydata(pts_y);
        ax[2].set_ydata(pts_ax);
        ax[3].set_ydata(pts_ay);
        f.canvas.draw();


