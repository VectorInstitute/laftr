import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
import sys

METRICS_FILE_NAME = 'test_metrics.csv'
figdir = '/u/madras/Projects/code/adv-fair-reps/test_figs/health_feb7'
errkey = 'Err'
dtemplate = '/ais/gobi5/madras/adv-fair-reps/experiments/{}/checkpoints'

dset_name = sys.argv[1]
d1_expname = sys.argv[2]
d1name = sys.argv[3]
d2_expname = sys.argv[4]
d2name = sys.argv[5]
d1 = dtemplate.format(d1_expname)
d2 = dtemplate.format(d2_expname)
if len(sys.argv) <= 6:
    dstring = '{}_vs_{}'.format(d1name, d2name)
    d_list = [(d1, d1name), (d2, d2name)]
else:
    d3_expname = sys.argv[6]
    d3name = sys.argv[7]
    d3 = dtemplate.format(d3_expname)
    if len(sys.argv) > 9:
        d4_expname = sys.argv[8]
        d4name = sys.argv[9]
        d4 = dtemplate.format(d4_expname)
        dexpname = sys.argv[10]
        d_list = [(d1, d1name), (d3, d3name), (d4, d4name), (d2, d2name)]
    else:
        dexpname = sys.argv[8]
        d_list = [(d1, d1name), (d3, d3name), (d2, d2name)]
    dstring = dexpname

# if dset_name == 'health':
#
#     # d1 = '/ais/gobi5/madras/adv-fair-reps/experiments/health_2aud_2rep_A_ckpts/checkpoints'
#     d1 = '/ais/gobi5/madras/adv-fair-reps/experiments/health_2aud_2rep_A_ckpts_fc1/checkpoints'
#
#     # d2 = '/ais/gobi5/madras/adv-fair-reps/experiments/health_eqodds_1rep_2aud_ckpts/checkpoints'
#     # d2 = '/ais/gobi5/madras/adv-fair-reps/experiments/health_eqodds_1rep_ckpts_2/checkpoints'
#     # d2 = '/ais/gobi5/madras/adv-fair-reps/experiments/health_dempar_1rep_ckpts_fc8/checkpoints'
#     # d2 = '/ais/gobi5/madras/adv-fair-reps/experiments/health_dempar_1rep_ckpts_fc2/checkpoints'
#
#     # d2 = '/ais/gobi5/madras/adv-fair-reps/experiments/health_dempar_2rep_fc1.5_ckpts/checkpoints'
#     # d2 = '/ais/gobi5/madras/adv-fair-reps/experiments/test_basic_health/checkpoints'
#     d2 = '/ais/gobi5/madras/adv-fair-reps/experiments/health_basic_dempar_wtd/checkpoints'
#
# elif dset_name == 'compas':
#     d1 = '/ais/gobi5/madras/adv-fair-reps/experiments/compas_2aud_2rep_A_ckpts_2/checkpoints'
#     d2 = '/ais/gobi5/madras/adv-fair-reps/experiments/compas_dempar_2rep_fc1.5_ckpts/checkpoints'
#     # d2 = '/ais/gobi5/madras/adv-fair-reps/experiments/test_basic_compas_2/checkpoints'
#
# d1name = 'eqoddsfc1'
# d2name = 'basicdempar_wtd'


res = {}


for phase in ['Valid', 'Test']:
    for fairkey in ['DI', 'DP', 'DI_FP']:
        plt.clf()
        for d, dnm in d_list:
            # print(d)
            # print(phase, fairkey, d1name)
            dnames = os.listdir(d)
            phase_dnames = filter(lambda s: phase in s, dnames)
            full_dnames = [os.path.join(d, s) for s in phase_dnames]
            errs = []
            fairs = []
            # print(full_dnames)
            for full_dname in full_dnames:
                fname = os.path.join(full_dname, METRICS_FILE_NAME)
                info = [s.strip().split(',') for s in open(fname, 'r').readlines()]
                info = {s[0]: float(s[1]) for s in info}
                try:
                    errs.append(info['ErrY'])
                    fairs.append(info[fairkey])
                except KeyError:
                    print('Could not find something in {}; moving on'.format(fname))
            plt.scatter(errs, fairs, label=dnm)
            res[(phase, fairkey, dnm)] = fairs
            res[(phase, errkey, dnm)] = errs

        plt.legend()
        plt.xlabel(errkey)
        plt.ylabel(fairkey)
        figname = os.path.join(figdir, '{}_{}_{}_{}.png'.format(dset_name, dstring, fairkey, phase))
        plt.savefig(figname)
        print('Saving to {}'.format(figname))

    # plt.clf()
    # plt.scatter(res[(phase, 'DI', d1name)], res[(phase, 'DP', d1name)], label=d1name)
    # plt.scatter(res[(phase, 'DI', d2name)], res[(phase, 'DP', d2name)], label=d2name)
    # plt.legend()
    # plt.xlabel('DI')
    # plt.ylabel('DP')
    # figname = os.path.join(figdir, '{}_{}_vs_{}_DI_vs_DP_{}.png'.format(dset_name, d1name, d2name, phase))
    # plt.savefig(figname)
    #
    # for dnm in [d1name, d2name]:
    #     slope, intercept, r_value, p_value, std_err = linregress(res[(phase, 'DI', dnm)], res[(phase, 'DP', dnm)])
    #     print('{}: Slope: {:.2f}, Intercept: {:.2f}'.format(dnm, slope, intercept))

