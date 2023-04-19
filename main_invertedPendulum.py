import pybullet_envs;
import gym;
from PPO_Agent import Agent;

import numpy as np;
import os;
import matplotlib.pyplot as plt;


#define the main function here
if __name__ == "__main__":

    env = gym.make('InvertedPendulumBulletEnv-v0');
    # env = gym.make('Pendulum-v1');
    
    # for trial_no in range(0,5):
    
    #     agent = Agent(env.observation_space.shape, env.action_space.shape, env.action_space.high[0], 2048, max_div=1e-2);

    #     # agent.visits = Visitation((tuple(env.observation_space.low),tuple(env.observation_space.high)), n_bins = 200);
    #     agent.state = env.reset();

    #     #before starting...create a directory for saving the agent..
    #     name = 'inverted_pendulum';
    #     trial_number = str(trial_no);
    #     filename = os.path.join('tmp','trpo_'+ name+'_trial'+trial_number);

    #     try:
    #         os.mkdir('tmp/trpo');
    #     except:
    #         print('tmp/ppo Already Exists. Overwriting the directory');


    #     #learn for 512000 steps i.e 250 rollout runs
    #     agent.learn(env, n_learn_steps=1e6, n_rollout_steps=2048);
        
    #     plt.plot(agent.training_logs['score']);
    #     # plt.plot(agent.training_logs['lagrangian']);
    #     agent.save_models();

    #     os.rename('tmp\\trpo', filename);

    #run the model
    agent = Agent(env.observation_space.shape, env.action_space.shape, env.action_space.high[0], 2048, max_div=1e-2);
    agent.load_models();
    
    
    env.render(mode = 'human');
    #run for predefined number of episodes
    for epi in range(100):
        observation = env.reset();env.render();
        done = False;
        score = 0;
        while not done:
            action, logprob = agent.choose_action(observation);

            observation_, reward, done, info = env.step(action);
            env.render();
            score+=reward;

            observation = observation_;
        print(f'episode: {epi}, score: {score:.4f}');


