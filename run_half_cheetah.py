import pybullet_envs;
import gym;
from PPO_Agent import Agent;
from utils import Visitation, plotConfInterval, linear_schedule;

import numpy as np;
import os;
import matplotlib.pyplot as plt;

if __name__ =="__main__":

    env = gym.make('HalfCheetahBulletEnv-v0');

    #run the model
    agent = Agent(env.observation_space.shape, env.action_space.shape, env.action_space.high[0], 2048, max_div=1e-2);
    agent.actor.checkpoint_file='tmp/trpo_helf_cheetah_trial41/actor_trpo';
    agent.critic.checkpoint_file='tmp/trpo_helf_cheetah_trial41/critic_trpo';
    agent.load_models();

    #run for predefined number of episodes
    env.render(mode = 'human');
    for epi in range(100):
        observation = env.reset();#env.render();
        done = False;
        score = 0; i=0;
        while not done:
            action, logprob = agent.choose_action(observation);

            observation_, reward, done, info = env.step(action);
            #env.render();
            score+=reward;

            observation = observation_;
            i+=1;
            if (i % 20 == 0):
                print('stopping here');
        print(f'episode: {epi}, score: {score:.4f}');