from get_best_epochs import num2str
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from codebase.metrics import *
from sklearn.manifold import TSNE
import sys

figdir = '/u/madras/Projects/code/adv-fair-reps/test_figs/ckpt_pred_histograms'
expdir5 = '/ais/gobi5/madras/adv-fair-reps/experiments'
expdir6 = '/ais/gobi6/madras/adv-fair-reps/experiments'


expnames = ['rand_save_DP_health_aud1_fc2.5_ua',
            'rand_save_DP_health_aud1_fc2.5_ep2000_ua_debug',
            'rand_save_DP-xent_health_aud1_fc2.5_ep2000_ua',
            'rand_save_DP-xent_health_aud1_fc0.5_ep2000_ua',
            'rand_save_DP_adult_aud1_fc2.5_ua',
            'rand_save_DP_adult_aud1_fc2.5_ep2000_ua_debug',
            'rand_save_DP-xent_adult_aud1_fc2.5_ep2000_ua',
            'rand_save_DP-xent_adult_aud1_fc0.5_ep2000_ua',
            'run_gradhist_ww_mn',
            'run_gradhist_xent_mn',
            'run_gradhist_hw_mn',]

epoch = 500
eps = 1e-8

for expname in expnames:
    print(expname)
    if os.path.exists(os.path.join(expdir5, expname)):
        expdir = expdir5
    else:
        expdir = expdir6


    ckpts = list(filter(lambda d: 'Valid' in d, os.listdir(os.path.join(expdir, expname, 'checkpoints'))))
    epochs = sorted([int(c.split('_')[1]) for c in ckpts])

    ckptdirs = [os.path.join(expdir, expname, 'checkpoints', 'Epoch_{:d}_Valid'.format(ep), 'npz') for ep in epochs]
    mean_ce_grads = []
    median_ce_grads = []

    for ckptdir in ckptdirs:
        pred_file = os.path.join(ckptdir, 'Y_hat.npz')
        target_file = os.path.join(ckptdir, 'Y.npz')
        preds = np.load(pred_file)['X']
        targets = np.load(target_file)['X']
        ce_grads = np.multiply(targets, 1. / (preds + eps)) + np.multiply(1. - targets, 1. / (1 - preds + eps))
        mean_ce_grads.append(np.mean(ce_grads))
        median_ce_grads.append(np.median(ce_grads))

    plt.clf()
    plt.plot(epochs, mean_ce_grads)
    figname = os.path.join(figdir, '{}_meanCEGrad_plot.png'.format(expname))
    plt.savefig(figname)

    plt.clf()
    plt.plot(epochs, median_ce_grads)
    figname = os.path.join(figdir, '{}_medianCEGrad_plot.png'.format(expname))
    plt.savefig(figname)


    ckptdir = os.path.join(expdir, expname, 'checkpoints', 'Epoch_{:d}_Valid'.format(epoch), 'npz')

    pred_file = os.path.join(ckptdir, 'Y_hat.npz')
    target_file = os.path.join(ckptdir, 'Y.npz')
    pred_sens_file = os.path.join(ckptdir, 'A_hat.npz')
    sens_file = os.path.join(ckptdir, 'A.npz')
    z_file = os.path.join(ckptdir, 'Z.npz')


    preds = np.load(pred_file)['X']
    targets = np.load(target_file)['X']
    pred_a = np.load(pred_sens_file)['X']
    a = np.load(sens_file)['X']
    z = np.load(z_file)['X']

    # print(preds.shape, targets.shape, a.shape, pred_a.shape)

    pr = PR(preds)
    base_pr = PR(targets)
    a_pr = PR(a)
    pred_a_pr = PR(pred_a)

    print('PR (pred): {:.3f}'.format(pr))
    print('PR (base): {:.3f}'.format(base_pr))
    print('PR (pred A): {:.3f}'.format(pred_a_pr))
    print('PR (base A): {:.3f}'.format(a_pr))


    err = errRate(targets, preds)
    nll = NLL(targets, preds)

    print('Error rate is: {:.3f}'.format(err))
    print('CE is: {:.3f}'.format(nll))
    print('DP is: {:.3f}'.format(DP(preds, a)))

    # print('Pred Y Hist')
    # for p in np.arange(0, 101, 10):
    #     print(p, np.percentile(preds, p))
        
    errA = errRate(a, pred_a)
    nllA = NLL(a, pred_a)
    l1A = np.mean(np.abs(pred_a - a))


    print('Error rate (A) is: {:.3f}'.format(errA))
    print('CE (A) is: {:.3f}'.format(nllA))
    print('L1 (A) is: {:.3f}'.format(l1A))

    # print('Pred A Hist')
    # for p in np.arange(0, 101, 10):
    #     print(p, np.percentile(pred_a, p))

    plt.clf()
    plt.hist(preds)
    figname = os.path.join(figdir, '{}_{:d}_pred_hist.png'.format(expname, epoch))
    plt.savefig(figname)

    plt.clf()
    plt.hist(pred_a)
    figname = os.path.join(figdir, '{}_{:d}_predA_hist.png'.format(expname, epoch))
    plt.savefig(figname)


    ce_grads = np.multiply(targets, 1. / (preds + eps)) + np.multiply(1. - targets, 1. / (1 - preds + eps))
    print(ce_grads.shape, targets.shape, preds.shape, max(ce_grads), min(ce_grads))
    plt.clf()
    plt.hist(np.log(ce_grads))
    figname = os.path.join(figdir, '{}_{:d}_CEGrad_hist.png'.format(expname, epoch))
    plt.savefig(figname)

    # print('Pred A Hist')
    # for p in np.arange(0, 101, 10):
    #     print(p, np.percentile(ce_grads, p))
    # print(sorted(ce_grads)[-20:])
    print('Mean CE grad', np.mean(ce_grads))
    # plt.clf()
    # plt.hist(pred_a)
    # figname = os.path.join(figdir, '{}_{:d}_predA_hist.png'.format(expname, epoch))
    # plt.savefig(figname)

    #
    # #get smaller sample of z
    # tsne_pct = 0.05
    # n = z.shape[0]
    # shuf = np.arange(n)
    # np.random.shuffle(shuf)
    # tsne_inds = shuf[:int(n * tsne_pct)]
    # z_tsne = z[tsne_inds]
    # z_embedded = TSNE(verbose=1).fit_transform(z_tsne)
    # print(z_embedded.shape, z_tsne.shape)
    # plt.clf()
    # colours = list(map(lambda y, a: {(0., 0.): 'b',
    #                                  (1., 0.): 'c',
    #                                 (0., 1.): 'r',
    #                                 (1., 1.): 'm'}[(y, a)], targets[tsne_inds].flatten(), a[tsne_inds].flatten()))
    # plt.scatter(z_embedded[:,0], z_embedded[:,1], c=colours)
    # figname = os.path.join(figdir, '{}_{:d}_z_tsne.png'.format(expname, epoch))
    # plt.savefig(figname)

