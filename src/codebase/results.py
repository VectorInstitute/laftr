import numpy as np
import os
import tensorflow as tf
from codebase.utils import make_dir_if_not_exist


class ResultLogger(object):
    def __init__(self, dname, saver=None):
        self.dname = dname
        make_dir_if_not_exist(self.dname)
        self.ckptdir = os.path.join(self.dname, 'checkpoints')
        make_dir_if_not_exist(self.ckptdir)
        self.npzdir = os.path.join(self.dname, 'npz')
        make_dir_if_not_exist(self.npzdir)
        self.saver = saver if not saver is None else tf.train.Saver()
        self.testcsv_name = os.path.join(self.dname, 'test_metrics.csv')
        self.testcsv = open(self.testcsv_name, 'w')

    def save_metrics(self, D):
        """save D (a dictionary of metrics: string to float) as csv"""
        for k in D:
            s = '{},{:.7f}\n'.format(k, D[k])
            self.testcsv.write(s)
        self.testcsv.close()
        print('Metrics saved to {}'.format(self.testcsv_name))

    def save_tensors(self, D):
        for k in D:
            fname = os.path.join(self.npzdir, '{}.npz'.format(k))
            np.savez(fname, X=D[k])
