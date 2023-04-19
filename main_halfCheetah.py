import pybullet_envs;
import gym;
from PPO_Agent import Agent;
from utils import Visitation, plotConfInterval, linear_schedule;

import numpy as np;
import os;
import matplotlib.pyplot as plt;


#define the main function here
if __name__ == "__main__":

    env = gym.make('HalfCheetahBulletEnv-v0');
    # env = gym.make('Pendulum-v1');
    
    # for trial_no in range(39,40):
    
    #     agent = Agent(env.observation_space.shape, env.action_space.shape, env.action_space.high[0], 2048, max_div=1e-2);

    #     # agent.visits = Visitation((tuple(env.observation_space.low),tuple(env.observation_space.high)), n_bins = 200);
    #     agent.state = env.reset();

    #     #before starting...create a directory for saving the agent..
    #     name = 'helf_cheetah';
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
    
    #define the parameter aray for the trials
    gamma_arr = [0.99,0.95,0.9,0.7,0.5];
    gae_lmbd_arr = [0.95, 0.99, 0.9, 0.7, 0.5];
    max_div_arr = [1e-3, 1e-2, 1e-1, 1];
    n_epochs_arr = [10];
    max_ls_steps_arr = [10];
    max_cg_iters_arr = [20];
    schedule_rate_arr = [25000, 245, 100, 200, 300, 400, 500];
    rollout_length_arr = [2048, 2500, 5000, 10000, 20000, 50000];
    n_learn_steps = 1e7;

    #now loop through
    trial_no = 45;
    for gae_lmbd in gae_lmbd_arr:
        for gamma in gamma_arr:
            for rollout_length in rollout_length_arr:
                for max_div in max_div_arr:
                    for n_epochs in n_epochs_arr:
                        for max_ls_steps in max_ls_steps_arr:
                            for max_cg_iters in max_cg_iters_arr:
                                for schedule_rate in schedule_rate_arr:
                                    
                                    # if((trial_no >= 45)&(trial_no <= 58)):
                                    #     trial_no += 1;
                                    #     continue;
                                    
                                    name = 'helf_cheetah';
                                    trial_number = str(trial_no);
                                    filename = os.path.join('tmp','trpo_'+ name+'_trial'+trial_number+'a');

                                    try:
                                        os.mkdir('tmp/trpo');
                                    except:
                                        print('tmp/ppo Already Exists. Overwriting the directory');
                                    
                                    
                                    #create a file...
                                    f = open('tmp/trpo/config.txt', 'w');
                                    f.write(f'gamma: {gamma} \r\n');
                                    f.write(f'gae_lmbd: {gae_lmbd} \r\n');
                                    f.write(f'max_div: {max_div} \r\n');
                                    f.write(f'n_epochs: {n_epochs} \r\n');
                                    f.write(f'max_ls_steps: {max_ls_steps} \r\n');
                                    f.write(f'max_cg_iters: {max_cg_iters} \r\n');
                                    f.write(f'beta linearly 10 to 1000 in: {schedule_rate} \r\n');
                                    f.write(f'rollout length: {rollout_length} \r\n');
                                    f.write(f'n learn steps: {n_learn_steps} \r\n');
                                    f.close();

                                    beta_sched = linear_schedule(0, schedule_rate, 10, 1e5);

                                    agent = Agent(env.observation_space.shape, env.action_space.shape, env.action_space.high[0],
                                    rollout_length, gamma=gamma, gae_lambda= gae_lmbd, max_div=max_div, n_epochs=n_epochs, max_ls_steps=max_ls_steps,
                                    max_cg_iters=max_cg_iters,beta_schedule=beta_sched);

                                    # agent.visits = Visitation((tuple(env.observation_space.low),tuple(env.observation_space.high)), n_bins = 200);
                                    agent.state = env.reset();

                                    #learn for 512000 steps i.e 250 rollout runs
                                    agent.learn(env, n_learn_steps=n_learn_steps, n_rollout_steps=rollout_length)

                                    agent.save_models();

                                    os.rename('tmp\\trpo', filename);
                                    
                                    trial_no += 1;

    #run the model
    # agent = Agent(env.observation_space.shape, env.action_space.shape, env.action_space.high[0], 2048, max_div=1e-2);
    # agent.load_models();

    #run for predefined number of episodes
    env.render(mode = 'human');
    for epi in range(100):
        observation = env.reset();#env.render();
        done = False;
        score = 0;
        while not done:
            action, logprob = agent.choose_action(observation);

            observation_, reward, done, info = env.step(action);
            #env.render();
            score+=reward;

            observation = observation_;
        print(f'episode: {epi}, score: {score:.4f}');

    #plot training scores for the trials
    starts=[11,16,21, 24, 29,33]; ends =[15,20,23, 28, 32, 33]; betas = [10,100,1000, 'annealed', 'annealed_max_sq_ratio','annealed_msr_c1'];
    cf = plt.figure(figsize=(32,18));
    ca = cf.subplots(nrows=2, ncols=2, sharex=True);
    color_array = ['b', 'r', 'g', 'k', 'm', None]; i=0;
    for st, en, beta in zip(starts, ends, betas):
        f = plt.figure(figsize=(16,9));
        axes = f.subplots(nrows=2, ncols=2, sharex=True);
        # axes[0,1].sharex(axes[1,0]);
        
        scores = None; kl_divs = None; sq_ratio_divs =None; improvements=None; grad_norms = None;
        scoreLength = 1000; rolloutLength = 2048;
        start = st; end = en; beta_is = beta;
        for trial_no in range(start, end+1):
            tn = str(trial_no);
            col = color_array[min(trial_no-start, 5)]
            name = 'helf_cheetah';
            filename = os.path.join('tmp','trpo_'+ name+'_trial'+tn);
            os.rename(filename, 'tmp\\trpo');
            agent = Agent(env.observation_space.shape, env.action_space.shape, env.action_space.high[0], 2048);
            agent.load_models();
            sc = np.array(agent.training_logs['score']);
            # if(agent.training_logs.__contains__('beta')):
            #     beta = np.array(agent.training_logs['betas']);
            kl_div = np.array(agent.training_logs['kl_divs']);
            sq_ratio_div = np.array(agent.training_logs['sq_ratio_div']);
            improvement = np.array(agent.training_logs['improvment']) if agent.training_logs.__contains__('improvment') else [];
            grad_norm = np.array(agent.training_logs['surr_loss_grad_norm']) if agent.training_logs.__contains__('surr_loss_grad_norm') else [];

            scores = sc.reshape(1,-1) if scores is None else np.vstack([scores, sc.reshape(1,-1)]);
            kl_divs = kl_div.reshape(1,-1) if kl_divs is None else np.vstack([kl_divs, kl_div.reshape(1,-1)]);
            sq_ratio_divs = sq_ratio_div.reshape(1,-1) if sq_ratio_divs is None else np.vstack([sq_ratio_divs, sq_ratio_div.reshape(1,-1)]);
            improvements = improvement.reshape(1,-1) if improvements is None else np.vstack([improvements, improvement.reshape(1,-1)]);
            grad_norms = grad_norm.reshape(1,-1) if grad_norms is None else np.vstack([grad_norms, grad_norm.reshape(1,-1)]);

            axes[0,0].plot(np.arange(sc.shape[0])*scoreLength, sc, color=col, label=f'score_trial_no_{tn}');
            axes[0,1].plot(np.arange(grad_norm.shape[0])*rolloutLength, grad_norm, color=col, linestyle='solid', label=f'sl_grad_norm_trial_no{tn}');
            axes[1,0].plot(np.arange(sq_ratio_div.shape[0])*rolloutLength, sq_ratio_div,color=col, linestyle='dashed', label=f'sq_ratio_div_trial_no_{tn}');
            axes[1,1].plot(np.arange(improvement.shape[0])*rolloutLength, improvement, color=col, label=f'improvement_trial_no_{tn}');
            os.rename('tmp\\trpo', filename);
        plt.show; axes[0,0].legend();axes[0,1].legend(); axes[1,0].legend(); axes[1,1].legend();
        # axes[0,0].title(f'beta_{beta_is}');
        plt.savefig(f'plots\\new_beta_{beta_is}.png');
        # a = input('Press any Key to Continue');

        #compute statistics for scores and others...
        # scores = np.array(scores); kl_divs=np.array(kl_divs); sq_ratio_divs = np.array(sq_ratio_divs);
        # improvements = np.array(improvements);
        sc_m = scores.mean(0); sc_std = scores.std(0);
        kld_m = kl_divs.mean(0); kld_std = kl_divs.std(0);
        sqrd_m = sq_ratio_divs.mean(0); sqrd_std = sq_ratio_divs.std(0);
        imp_m = improvements.mean(0); imp_std = improvements.std(0);
        sl_gn_m = grad_norms.mean(0); sl_gn_std = grad_norms.std(0);
        col = color_array[i];
        plotConfInterval(np.arange(sc.shape[0])*scoreLength, sc_m, sc_std, ca[0,0], label=f'score_beta_{beta_is}', color=col);
        plotConfInterval(np.arange(kl_div.shape[0])*rolloutLength, sl_gn_m, sl_gn_std, ca[0,1], label=f'sl_grad_norm_beta_{beta_is}', color=col);
        plotConfInterval(np.arange(sq_ratio_div.shape[0])*rolloutLength, sqrd_m, sqrd_std, ca[1,0], label=f'sq_ratio_divergence_beta_{beta_is}', color=col);
        plotConfInterval(np.arange(improvement.shape[0])*rolloutLength, imp_m, imp_std, ca[1,1], label=f'improvement_beta_{beta_is}', color=col);
        # a = input('Press any Key to Continue');
        i+=1;
    cf.savefig( 'plots\\testSave7.png');
    print('Done');
    
