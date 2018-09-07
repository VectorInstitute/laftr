from find_test_result import BIG, get_ckpt_stats, loss
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

def num2str(x):
    if np.isclose(x % 1, 0.):
        return '{:d}'.format(int(x))
    else:
        return '{:.1f}'.format(x)


fm = 'eqopp'
if fm == 'eqopp':
    imgdir = '/u/madras/Projects/code/adv-fair-reps/test_figs/xent_vs_whw_eqopp_3'
    gammas_xent = np.arange(1, 9)
    gammas_whw = np.concatenate([np.arange(1, 4.5, 0.5), np.arange(5, 9, 0.5)])
    adult_name = 'adult_eqopp_{}_fc{}'
    health_name = 'health_eqopp_{}_fc{}_ua'
    fm_abbrev = 'DI_FP'
elif fm == 'dempar':
    imgdir = '/u/madras/Projects/code/adv-fair-reps/test_figs/xent_vs_whw_2'
    gammas_xent = np.arange(1, 9)
    gammas_whw = np.arange(1, 4.5, 0.5)
    adult_name = 'adult_dempar_{}_fc{}'
    health_name = 'health_dempar_{}_fc{}_ua'
    fm_abbrev = 'DP'

# imgdir = '/u/madras/Projects/code/adv-fair-reps/test_figs/xent_vs_whw_eqopp'
ckptdir = '/ais/gobi5/madras/adv-fair-reps/experiments'

#2 hyperparameters: lambda (validation fairness) and gamma (training fairness)
name_xent = 'xent'
name_whw = 'whw'
lamdas = np.arange(0.1, 2.1, 0.1)
mkr_xent = 's'
mkr_whw = 'o'


#for datasets in adult and health
for dset, dname_template_base in [('adult', adult_name), ('health', health_name)]:
    print(dset)
    for lamda in lamdas:
        print('Lambda={}'.format(num2str(lamda)))

        plt.clf()
        for gammas, model_name, mkr in [(gammas_xent, name_xent, mkr_xent), (gammas_whw, name_whw, mkr_whw)]:
            results = []
            print('\t{}'.format(model_name))
            if model_name == name_whw and fm == 'eqopp':
                dname_template = dname_template_base + '_2'
            else:
                dname_template = dname_template_base
            for gamma in gammas:

                bigdir = os.path.join(ckptdir, dname_template.format(model_name, num2str(gamma)), 'checkpoints')

                # loop through all validation directories
                ckpt_names = os.listdir(bigdir)
                valid_ckpt_names = filter(lambda s: 'Valid' in s, ckpt_names)

                best_loss = BIG
                best_epoch = -1
                best_err = BIG
                best_dp = BIG

                for d in valid_ckpt_names:
                    dname = os.path.join(bigdir, d)
                    stats = get_ckpt_stats(dname)
                    l, err, dp, _, _, _ = loss(stats, lamda, fm_abbrev, unsup=False)
                    if l < best_loss:
                        best_loss, best_err, best_dp = l, err, dp
                        ep = int(d.split('_')[1])
                        best_epoch = ep
                test_dname = 'Epoch_{:d}_Test'.format(best_epoch)
                test_resdir = os.path.join(bigdir, test_dname)
                test_stats = get_ckpt_stats(test_resdir)
                test_l, test_err, test_dp, _, _, _ = loss(test_stats, lamda, fm_abbrev, unsup=False)
                results.append([test_err, test_dp])
                print('\t\tGamma={}, Err: {:.4f}, {}: {:.4f}'.format(num2str(gamma), test_err, fm_abbrev, test_dp))

            plt.scatter([r[0] for r in results], [r[1] for r in results], label=model_name, marker=mkr, s=100)
        plt.legend()
        plt.xlabel('Error')
        plt.ylabel(fm_abbrev)
        figname = os.path.join(imgdir, '{}_lambda={}.png'.format(dset, num2str(lamda)))
        plt.savefig(figname)


#make a new plot for each lambda

#for xent and whw

#for each gamma

#find optimal setting

#save result in list


#plot xent vs whw for this lambda

