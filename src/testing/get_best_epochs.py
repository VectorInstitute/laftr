import numpy as np
import os
from testing.find_test_result import get_ckpt_stats, loss

BIG = 99999.

#def num2str(x):
#    if np.isclose(x % 1, 0.):
#        return '{:d}'.format(int(x))
#    else:
#        return '{:.1f}'.format(x)

def num2str(x, digits=None):
    if digits is None:
        if np.isclose(x % 1, 0.):
            return '{:d}'.format(int(x))
        else:
            i = 1
            while not np.isclose(x * (10 ** i) % 1, 0.) and i < 8:
                i += 1
            format_str = '{:.' + '{:d}'.format(i) + 'f}'
            return format_str.format(x)
    else:
        format_str = '{:.' + '{:d}'.format(digits) + 'f}'
        return format_str.format(x)

def get_best_epoch(expdir, fairkey, lamda):
    bigdir = os.path.join(expdir, 'checkpoints')

    # loop through all validation directories
    ckpt_names = os.listdir(bigdir)
    valid_ckpt_names = filter(lambda s: 'Valid' in s, ckpt_names)

    best_loss = BIG
    best_epoch = -1
    best_err = BIG
    best_dp = BIG
    # print(fairkey)
    for d in valid_ckpt_names:
        dname = os.path.join(bigdir, d)

        stats = get_ckpt_stats(dname)
        l, err, fair, _, _, _ = loss(stats, lamda, fairkey, unsup=False)
        if l < best_loss:
            ep = int(d.split('_')[1])
            # print(ep, l, err, fair)
            best_loss, best_err, best_fair = l, err, fair
            best_epoch = ep

    return best_epoch, best_fair

def write_best_epochs(fname, best_eps, lamda):
    f = open(fname, 'w')
    f.write('Lamda,{:.3f}\n'.format(lamda))
    for fm in sorted(best_eps):
        if np.isnan(best_eps[fm][0]): print(fname, fm, 'is nan')
        # print('{},{:d}\n'.format(fm, best_eps[fm]))
        f.write('{},{:d},{:.5f}\n'.format(fm, best_eps[fm][0], best_eps[fm][1]))
    f.close()

if __name__ == '__main__':
    # expdirs = ['/ais/gobi5/madras/adv-fair-reps/experiments/health_eqopp_whw_fc6_ua_2']
    expdir = '/ais/gobi5/madras/adv-fair-reps/experiments/'
    dp_dirs = ['adult_dempar_whw_fc{}'.format(num2str(gamma)) \
               for gamma in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]] + \
             ['health_dempar_whw_fc{}_ua'.format(num2str(gamma)) \
              for gamma in [1, 1.5, 2, 2.5, 3, 3.5, 4]]
    eqodds_dirs = ['adult_eqodds_whw_fc{}_2'.format(num2str(gamma)) \
               for gamma in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]] + \
                    ['health_eqodds_whw_fc{}_ua_2'.format(num2str(gamma)) \
                     for gamma in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]]
    eqopp_dirs = ['adult_eqopp_whw_fc{}_2'.format(num2str(gamma)) \
               for gamma in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]] + \
                    ['health_eqopp_whw_fc{}_ua_2'.format(num2str(gamma)) \
                     for gamma in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]]
    expdirs = [os.path.join(expdir, d) for d in dp_dirs + eqodds_dirs + eqopp_dirs]
    fmets = ['DP', 'DI', 'DI_FP', 'ErrA']
    lamda = 1.
    # somehow loop over all runs
    for d in expdirs:
        # for each fairness metric
        best_eps = {}
        for fm in fmets:
            best_ep, best_fair = get_best_epoch(d, fm, lamda)
            best_eps[fm] = (best_ep, best_fair)
        fname = os.path.join(d, 'best_validation_fairness.txt')
        write_best_epochs(fname, best_eps, lamda)
        print('Wrote metrics to {}'.format(fname))




    # get reprs from best epoch

    # run accuracy-only mlp

    # record the fairness metrics


    #TODO: Re-run Adult with sensitive attribute
    #TODO: Re-run transfer stuff with whw
    #TODO: Try comparing xent with whw on transfer????
