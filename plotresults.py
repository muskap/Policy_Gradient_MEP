import pybullet_envs;
import gym;
from PPO_Agent import Agent;
from utils import Visitation, plotConfInterval, linear_schedule;


import numpy as np;
import os;
import matplotlib.pyplot as plt;
import pickle as pkl;




if __name__ == "__main__":

    env = gym.make('HalfCheetahBulletEnv-v0');

    starts=[11,16,21, 24]; ends =[15,20,23, 28]; betas = [10,100,1000, 'annealed'];
    # starts.append(29); ends.append(32); betas.append('annealed_max_sq_ratio');
    # starts.append(33); ends.append(33); betas.append('annealed_msr_c1');
    starts.append(40); ends.append(45); betas.append('annealed_sq_ratio_lim_1e-3');
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
            # filename = os.path.join('tmp','trpo_'+ name+'_trial'+tn);
            filename = 'tmp/trpo_'+ name+'_trial'+tn+'/agent_training_logs.pkg';
            # os.rename(filename, 'tmp\\trpo');
            # agent = Agent(env.observation_space.shape, env.action_space.shape, env.action_space.high[0], 2048);
            f = open(filename, 'rb');
            logs = pkl.load(f);
            f.close();
            sc = np.array(logs['score']);
            # if(agent.training_logs.__contains__('beta')):
            #     beta = np.array(agent.training_logs['betas']);
            kl_div = np.array(logs['kl_divs']);
            sq_ratio_div = np.array(logs['sq_ratio_div']);
            improvement = np.array(logs['improvment']) if logs.__contains__('improvment') else [];
            grad_norm = np.array(logs['surr_loss_grad_norm']) if logs.__contains__('surr_loss_grad_norm') else [];

            scores = sc.reshape(1,-1) if scores is None else np.vstack([scores, sc.reshape(1,-1)]);
            kl_divs = kl_div.reshape(1,-1) if kl_divs is None else np.vstack([kl_divs, kl_div.reshape(1,-1)]);
            sq_ratio_divs = sq_ratio_div.reshape(1,-1) if sq_ratio_divs is None else np.vstack([sq_ratio_divs, sq_ratio_div.reshape(1,-1)]);
            improvements = improvement.reshape(1,-1) if improvements is None else np.vstack([improvements, improvement.reshape(1,-1)]);
            grad_norms = grad_norm.reshape(1,-1) if grad_norms is None else np.vstack([grad_norms, grad_norm.reshape(1,-1)]);

            axes[0,0].plot(np.arange(sc.shape[0])*scoreLength, sc, color=col, label=f'score_trial_no_{tn}');
            axes[0,1].plot(np.arange(grad_norm.shape[0])*rolloutLength, grad_norm, color=col, linestyle='solid', label=f'sl_grad_norm_trial_no{tn}');
            axes[1,0].plot(np.arange(sq_ratio_div.shape[0])*rolloutLength, sq_ratio_div,color=col, linestyle='dashed', label=f'sq_ratio_div_trial_no_{tn}');
            axes[1,1].plot(np.arange(improvement.shape[0])*rolloutLength, improvement, color=col, label=f'improvement_trial_no_{tn}');
            # os.rename('tmp\\trpo', filename);
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
        col = color_array[min(i,5)];
        plotConfInterval(np.arange(sc.shape[0])*scoreLength, sc_m, sc_std, ca[0,0], label=f'score_beta_{beta_is}', color=col);
        plotConfInterval(np.arange(kl_div.shape[0])*rolloutLength, sl_gn_m, sl_gn_std, ca[0,1], label=f'sl_grad_norm_beta_{beta_is}', color=col);
        plotConfInterval(np.arange(sq_ratio_div.shape[0])*rolloutLength, sqrd_m, sqrd_std, ca[1,0], label=f'sq_ratio_divergence_beta_{beta_is}', color=col);
        plotConfInterval(np.arange(improvement.shape[0])*rolloutLength, imp_m, imp_std, ca[1,1], label=f'improvement_beta_{beta_is}', color=col);
        # a = input('Press any Key to Continue');
        ncf = plt.figure(figsize=(32,18));
        nca = ncf.subplots(nrows=1, ncols=1);
        plotConfInterval(np.arange(sc.shape[0]), sc_m, sc_std, nca, color=col);
        nca.set_xlabel('iterations',fontdict={'size':'xx-large'});
        nca.set_ylabel('-1 x score',fontdict={'size':'xx-large'});
        nca.set_title(f'score_beta_{beta_is}',fontdict={'size':'xx-large'});
        plt.savefig(f'plots\\scoredist_new_beta_{beta_is}.png');
        i+=1;
    cf.savefig( 'plots\\testSave8.png');
    print('Done');