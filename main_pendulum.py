import pybullet_envs;
import gym;
from PPO_Agent import Agent;
from utils import Visitation;

import numpy as np;
import os;
import matplotlib.pyplot as plt;


#define the main function here
if __name__ == "__main__":

    env = gym.make('Pendulum-v1');
    # env = gym.make('InvertedPendulumBulletEnv-v0');

    for trial_no in range(0,2):
    
        agent = Agent(env.observation_space.shape, env.action_space.shape, env.action_space.high[0], 2048, max_div=1e-2);

        agent.visits = Visitation((tuple(env.observation_space.low),tuple(env.observation_space.high)), n_bins = 200);
        agent.state = env.reset();

        #before starting...create a directory for saving the agent..
        name = 'pendulum';
        trial_number = str(trial_no);
        filename = os.path.join('tmp','trpo_'+ name+'_trial'+trial_number);

        try:
            os.mkdir('tmp/trpo');
        except:
            print('tmp/ppo Already Exists. Overwriting the directory');


        #learn for 512000 steps i.e 250 rollout runs
        agent.learn(env, n_learn_steps=512000, n_rollout_steps=2048);
        
        plt.plot(agent.training_logs['score']);
        # plt.plot(agent.training_logs['lagrangian']);
        agent.save_models();

        os.rename('tmp\\trpo', filename);

    #run the model
    # agent = Agent(env.observation_space.shape, env.action_space.shape, env.action_space.high[0], 2048, max_div=1e-2);
    # agent.load_models();

    #run for predefined number of episodes
    env.reset();
    env.render(mode = 'human');
    for epi in range(100):
        observation = env.reset();#env.render();
        done = False;
        score = 0;
        while not done:
            action, logprob = agent.choose_action(observation);

            observation_, reward, done, info = env.step(action);
            env.render();
            score+=reward;

            observation = observation_;
        print(f'episode: {epi}, score: {score:.4f}');


