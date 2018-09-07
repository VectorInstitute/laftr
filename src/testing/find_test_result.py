import argparse
import os

BIG = 99999.

def get_ckpt_stats(d):
    fname = os.path.join(d, 'test_metrics.csv')
    info = [l.strip().split(',') for l in open(fname).readlines()]
    info_d = {l[0]: float(l[1]) for l in info}
    return info_d

def loss(stats, coeff, key, unsup):
    if not unsup:
        err = stats['ErrY']
    else:
        err = stats['Recon']
    di = stats['DI']
    dp = stats['DP']
    difp = stats['DI_FP']

    fair = stats[key]
    if key == 'ErrA':
        fair = -fair
    # print(err, fair, coeff)
    # print(type(err), type(fair), type(coeff))

    l = err + coeff * fair
    return l, err, fair, di, dp, difp

def get_filter_function(filters):
    def f(D):
        res = True
        for filt in filters:
            if len(filt) > 1:
                k = filt[0];  direction = filt[1]; val = float(filt[2])
                if not ((D[k] <= val and direction == 'lt') or (D[k] >= val and direction == 'gt')):
                    res = False
        return res
    return f

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='find test results')
    parser.add_argument('-fc', '--fair_coef', help='coefficient on fairness (auditor/adversarial) term', default='1')
    parser.add_argument('-k', '--key', help='use DI or DP?', default='DI')
    parser.add_argument('-u', '--unsup', help='unsupervised?', default='False')
    parser.add_argument('-n', '--name', help='experiment name', default='temp')
    parser.add_argument('-filt', '--filters', help='filters', default='') #colon separated, comma separated lists
    parser.add_argument('-ed', '--exp_dir', help='experiment directory', default='/ais/gobi5/madras/adv-fair-reps/experiments')
    parser.add_argument('--local', dest='gobi', help='run local', action='store_false')
    parser.set_defaults(gobi=True)
    args = vars(parser.parse_args())
    #need experiment name
    expname = args['name']
    if args['exp_dir'] == 'gobi4':
        args['exp_dir'] = '/ais/gobi4/madras/adv-fair-reps/experiments'

    expdir = args['exp_dir']
    #and fairness coefficient
    fair_coeff = float(args['fair_coef'])
    fairkey = args['key']
    unsup = (args['unsup'] == 'True')

    supkey  = 'Err' if not unsup else 'Recon'

    filters = [l.split(',') for l in args['filters'].split(':')]
    # print(filters)
    filt = get_filter_function(filters)

    bigdir = os.path.join(expdir, expname, 'checkpoints')

    #loop through all validation directories
    ckpt_names = os.listdir(bigdir)
    valid_ckpt_names = filter(lambda s: 'Valid' in s, ckpt_names)

    best_loss = BIG
    best_epoch = -1
    best_err = BIG
    best_fair = BIG
    best_di = BIG
    best_dp = BIG
    best_difp = BIG

    for d in valid_ckpt_names:
        dname = os.path.join(bigdir, d)
        stats = get_ckpt_stats(dname)
        if filt(stats):
            l, err, fair, di, dp, difp = loss(stats, fair_coeff, fairkey, unsup)
            if l < best_loss:
                best_loss, best_err, best_fair, best_di, best_dp, best_difp = l, err, fair, di, dp, difp
                ep = int(d.split('_')[1])
                best_epoch = ep

    test_dname = 'Epoch_{:d}_Test'.format(best_epoch)
    test_resdir = os.path.join(bigdir, test_dname)
    test_stats = get_ckpt_stats(test_resdir)
    test_l, test_err, test_fair, test_di, test_dp, test_difp = loss(test_stats, fair_coeff, fairkey, unsup)
    msg = 'Best validation epoch was {:d}: Valid loss: {:.3f}, Valid {} {:.3f}, Valid DI {:.3f}, Valid DP {:.3f}, Valid DIFP {:.3f}, Valid Fair {:.3f} ||\
                            Test loss: {:.3f}, Test {} {:.3f}, Test DI {:.3f}, Test DP {:3f}, Test DIFP {:.3f}, Test Fair {:.3f}'.format(best_epoch, best_loss, supkey, best_err, best_di, best_dp, best_difp, best_fair, \
                                                                                                 test_l, supkey, test_err, test_di, test_dp, test_difp, test_fair)
    print(msg)