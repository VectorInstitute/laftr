from codebase.metrics import *
import numpy as np


BIG_EPOCH = 1000000


class Tester(object):
    def __init__(self, model, data, sess, reslogger):
        self.data = data
        if not self.data.loaded:
            self.data.load()
            self.data.make_validation_set()
        self.model = model
        self.sess = sess
        self.reslogger = reslogger

    def evaluate(self, mb_size, phase='test', save=True):
        # for mb in training data
        test_iter = self.data.get_batch_iterator(phase, mb_size)
        test_L = {'class': 0., 'disc': 0., 'class_err': 0., 'disc_err': 0., 'recon': 0}
        num_batches = 0
        Y_hats = np.empty((0, 1))
        A_hats = np.empty((0, 1))
        Zs = np.empty((0, self.model.zdim))
        Ys = np.empty((0, 1))
        As = np.empty((0, 1))

        for x, y, a in test_iter:
            num_batches += 1
            if len(x) < mb_size:  # hack for WGAN-GP training; don't process weird-sized batches
                continue

            # make feed dict
            feed_dict = {self.model.X: x, self.model.Y: y, self.model.A: a, self.model.epoch: np.array([BIG_EPOCH])}

            # run forward encoder-classifier-decoder
            class_loss, recon_loss, class_err, Y_hat, Z, Y = self.sess.run(
                [self.model.class_loss,
                 self.model.recon_loss,
                 self.model.class_err,
                 self.model.Y_hat,
                 self.model.Z,
                 self.model.Y],
                feed_dict=feed_dict
            )
            # run forward auditor
            aud_loss, aud_err, total_loss, A_hat, A = self.sess.run(
                [self.model.aud_loss,
                 self.model.aud_err,
                 self.model.loss,
                 self.model.A_hat,
                 self.model.A],
                feed_dict=feed_dict)

            test_L['class'] += np.mean(class_loss)
            test_L['disc'] += np.mean(aud_loss)
            test_L['class_err'] += class_err
            test_L['disc_err'] += aud_err
            test_L['recon'] += np.mean(recon_loss)

            Y_hats = np.concatenate((Y_hats, Y_hat))
            Zs = np.concatenate((Zs, Z))
            Ys = np.concatenate((Ys, Y))
            A_hats = np.concatenate((A_hats, A_hat))
            As = np.concatenate((As, A))

        Y_hat = Y_hats
        Y = Ys
        A_hat = A_hats
        A = As
        Z = Zs

        tensorD = {}
        tensorD['Y_hat'] = Y_hat
        tensorD['Z'] = Z
        tensorD['Y'] = Y
        tensorD['A_hat'] = A_hat
        tensorD['A'] = A

        for d in tensorD:
            print(tensorD[d].shape)

        metD = {}

        for k in test_L: test_L[k] /= num_batches
        test_L['ttl'] = test_L['class'] - test_L['disc']
        test_res_str = 'Test score: Class CE: {class:.3f}, Disc CE: {disc:.3f}, Ttl CE: {ttl:.3f},' + \
                       ' Class Err: {class_err:.3f} Disc Err: {disc_err:.3f}'
        print(test_res_str.format(**test_L))

        metD['ClassCE'] = test_L['class']
        metD['DiscCE'] = test_L['disc']
        metD['TtlCE'] = test_L['ttl']

        err = errRate(Y, Y_hat)
        difp = DI_FP(Y, Y_hat, A)
        difn = DI_FN(Y, Y_hat, A)
        di = DI(Y, Y_hat, A)
        err_a = errRate(A, A_hat)
        dp = DP(Y_hat, A)

        metrics_str = 'Error Rate: {:.3f},  DI: {:.3f}, di_FP: {:.3f}, di_FN: {:.3f}'.format(err, di, difp, difn) \
                    + '\nError Rate (A): {:.3f}'.format(err_a)
        print(metrics_str)

        metD['ErrY'] = err
        metD['DI'] = di
        metD['DI_FP'] = difp
        metD['DI_FN'] = difn
        metD['ErrA'] = err_a
        metD['DP'] = dp
        metD['Recon'] = test_L['recon']
        errMaskA = np.abs(A - A_hat)

        print('\nPredicting Y')
        for mask, mask_nm in [(np.ones_like(A), 'A=?'), (A, 'A=1'), (1 - A, 'A=0'), (1 - errMaskA, 'ACor'), (errMaskA, 'AWro')]:
            pr_base = subgroup(PR, mask, Y)
            nr_base = subgroup(NR, mask, Y)
            pr = subgroup(PR, mask, Y_hat)
            nr = subgroup(NR, mask, Y_hat)
            tpr = subgroup(TPR, mask, Y, Y_hat)
            fpr = subgroup(FPR, mask, Y, Y_hat)
            tnr = subgroup(TNR, mask, Y, Y_hat)
            fnr = subgroup(FNR, mask, Y, Y_hat)
            err = subgroup(errRate, mask, Y, Y_hat)

            s = '{}: Base PR: {:.3f}, Base NR: {:.3f},  PR: {:.3f}, NR: {:.3f}, Err: {:.3f}, '.format(mask_nm, pr_base, nr_base, pr, nr, err) \
                + 'TPR: {:.3f}, FNR: {:.3f}, TNR: {:.3f}, FPR: {:.3f}'.format(tpr, fnr, tnr, fpr)
            print(s)

            metD['{}_BasePR'.format(mask_nm)] = pr_base
            metD['{}_BaseNR'.format(mask_nm)] = nr_base
            metD['{}_PredPR'.format(mask_nm)] = pr
            metD['{}_PredNR'.format(mask_nm)] = nr
            metD['{}_Error'.format(mask_nm)] = err
            metD['{}_TPR'.format(mask_nm)] = tpr
            metD['{}_TNR'.format(mask_nm)] = tnr
            metD['{}_FPR'.format(mask_nm)] = fpr
            metD['{}_FNR'.format(mask_nm)] = fnr

        print('\nPredicting A')
        errMaskY = np.abs(Y - Y_hat)

        for mask, mask_nm in [(np.ones_like(A), 'Y=?'), (Y, 'Y=1'), (1 - Y, 'Y=0'), (1 - errMaskY, 'YCor'), (errMaskY, 'YWro')]:
            pr_base = subgroup(PR, mask, A)
            nr_base = subgroup(NR, mask, A)
            pr = subgroup(PR, mask, A_hat)
            nr = subgroup(NR, mask, A_hat)
            tpr = subgroup(TPR, mask, A, A_hat)
            fpr = subgroup(FPR, mask, A, A_hat)
            tnr = subgroup(TNR, mask, A, A_hat)
            fnr = subgroup(FNR, mask, A, A_hat)
            err = subgroup(errRate, mask, A, A_hat)

            s = '{}: Base-A1-rate: {:.3f}, Base-A0-rate: {:.3f}, Pred A1-rate: {:.3f}, Pred-A0-rate: {:.3f}, Err: {:.3f}, '.format(mask_nm, pr_base, nr_base, pr, nr, err) \
                + 'A1-Correct: {:.3f}, A1-Wrong: {:.3f}, A0-Correct: {:.3f}, A0-Wrong: {:.3f}'.format(tpr, fnr, tnr, fpr)
            print(s)

            metD['{}_Base-A1-rate'.format(mask_nm)] = pr_base
            metD['{}_Base-A0-rate'.format(mask_nm)] = nr_base
            metD['{}_Pred-A1-rate'.format(mask_nm)] = pr
            metD['{}_Pred-A0-rate'.format(mask_nm)] = nr
            metD['{}_Error'.format(mask_nm)] = err
            metD['{}_A1-Correct'.format(mask_nm)] = tpr
            metD['{}_A1-Wrong'.format(mask_nm)] = fnr
            metD['{}_A0-Correct'.format(mask_nm)] = tnr
            metD['{}_A0-Wrong'.format(mask_nm)] = fpr

        self.reslogger.save_metrics(metD)
        if save:
            self.reslogger.save_tensors(tensorD)
