from find_test_result import BIG, get_ckpt_stats, loss
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def num2str(x):
    if np.isclose(x % 1, 0.):
        return '{:d}'.format(int(x))
    else:
        return '{:.1f}'.format(x)


imgdir = '/u/madras/Projects/code/adv-fair-reps/test_figs/whw_sep'
ckptdir = '/ais/gobi5/madras/adv-fair-reps/experiments'

#2 hyperparameters: lambda (validation fairness) and gamma (training fairness)
gammas = np.arange(1, 4.5, 0.5)
name_dp = 'dempar'
name_eqodds = 'eqodds'
name_eqopp = 'eqopp'
lamdas = np.arange(0.1, 2.1, 0.1)
mkr_dp = 's'
mkr_eqodds = 'o'
mkr_eqopp = 'v'
key_dp = 'DP'
key_eqodds = 'DI'
key_eqopp = 'DI_FP'
fairkeys = [key_dp, key_eqodds, key_eqopp]

#for datasets in adult and health
for dset, dname_template in [('adult', 'adult_{}_whw_fc{}'), ('health', 'health_{}_whw_fc{}_ua')]:
    print(dset)
    for bigfairkey in fairkeys:
        for lamda in lamdas:
            print('Lambda={}'.format(num2str(lamda)))

            plt.clf()
            # for model_name, mkr in [(name_dp, mkr_dp), (name_eqodds, mkr_eqodds), (name_eqopp, mkr_eqopp)]:
            for model_name, mkr, fairkey in [(name_dp, mkr_dp, key_dp), (name_eqodds, mkr_eqodds, key_eqodds), (name_eqopp, mkr_eqopp, key_eqopp)]:

                results = []
                for gamma in gammas:
                    print('\tGamma={}'.format(num2str(gamma)))
                    dname_complete = dname_template.format(model_name, num2str(gamma)) + ('' if model_name == name_dp else '_2')
                    bigdir = os.path.join(ckptdir, dname_complete, 'checkpoints')

                    # loop through all validation directories
                    ckpt_names = os.listdir(bigdir)
                    valid_ckpt_names = filter(lambda s: 'Valid' in s, ckpt_names)

                    best_loss = BIG
                    best_epoch = -1
                    best_err = BIG
                    best_fair = BIG

                    for d in valid_ckpt_names:
                        dname = os.path.join(bigdir, d)
                        stats = get_ckpt_stats(dname)
                        l, err, fair, _, _, _ = loss(stats, lamda, fairkey, unsup=False)
                        if l < best_loss:
                            best_loss, best_err, best_fair = l, err, fair
                            ep = int(d.split('_')[1])
                            best_epoch = ep
                    test_dname = 'Epoch_{:d}_Test'.format(best_epoch)
                    test_resdir = os.path.join(bigdir, test_dname)
                    test_stats = get_ckpt_stats(test_resdir)
                    test_l, test_err, test_fair, _, _, _ = loss(test_stats, lamda, bigfairkey, unsup=False)
                    results.append([test_err, test_fair])
                plt.scatter([r[0] for r in results], [r[1] for r in results], label=model_name, marker=mkr, s=100)
            plt.legend()
            plt.xlabel('Error')
            plt.ylabel(bigfairkey)
            figname = os.path.join(imgdir, '{}_{}_lambda={}.png'.format(dset, bigfairkey, num2str(lamda)))
            plt.savefig(figname)

