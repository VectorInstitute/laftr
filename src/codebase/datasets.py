import numpy as np
import collections
from codebase.metrics import *

def mb_round(t, bs):
    """take array t and batch_size bs and trim t to make it divide evenly by bs"""
    new_length = len(t) // bs * bs
    return t[:new_length, :]
 

class Dataset(object):
    def __init__(self, name, attr0_name, attr1_name, npzfile, seed=0, use_attr=False, load_on_init=True, y2i=None, pred_attr=False, batch_size=None, **kwargs):
        self.name = name
        self.attr0_name = attr0_name
        self.attr1_name = attr1_name
        self.npzfile = npzfile
        self.use_attr = use_attr
        self.pred_attr = pred_attr
        self.batch_size = batch_size

        self.loaded = False
        self.seed = seed
        self.y2i = y2i
        if load_on_init:
            self.load()
            self.make_validation_set()

   
    def load(self):
        if not self.loaded:
            dat = np.load(self.npzfile)
            self.dat = dat
            self.x_train = dat['x_train']
            self.x_test = dat['x_test']
            self.attr_train = dat['attr_train']
            self.attr_test = dat['attr_test']
            print('y shape', dat['y_train'].shape)
            if dat['y_train'].shape[1] > 1:
                print('changing shape')
                self.y_train = np.expand_dims(dat['y_train'][:,1], 1)
                self.y_test = np.expand_dims(dat['y_test'][:,1], 1)
            else:
                self.y_train = dat['y_train']
                self.y_test = dat['y_test']

            if self.pred_attr:
                self.y_train = self.attr_train
                self.y_test = self.attr_test

            # get valid inds
            if 'valid_inds' in dat:
                self.train_inds = dat['train_inds']
                self.valid_inds = dat['valid_inds']
            if 'y2_train' in dat:
                self.y2_train = dat['y2_train']
                self.y2_test = dat['y2_test']

            if 'x_valid' in dat:
                self.x_valid = dat['x_valid']
                self.y_valid = np.expand_dims(dat['y_valid'][:,1], 1)
                self.attr_valid = dat['attr_valid']

            if not self.y2i is None:

                print('using feature {:d}'.format(self.y2i))
                self.y_train = np.expand_dims(self.y2_train[:,self.y2i], 1)
                self.y_test = np.expand_dims(self.y2_test[:, self.y2i], 1)

            if self.use_attr:
                self.x_train = np.concatenate([self.x_train, self.attr_train], 1)
                self.x_test = np.concatenate([self.x_test, self.attr_test], 1)
                if 'x_valid' in dat:
                    self.x_valid = np.concatenate([dat['x_valid'], self.attr_valid], 1)
            self.loaded = True

    def make_validation_set(self, force=False):
        if not hasattr(self, 'x_valid') or force:
            self.x_valid = self.x_train[self.valid_inds]
            self.y_valid = self.y_train[self.valid_inds]
            self.attr_valid = self.attr_train[self.valid_inds]

            self.x_train = self.x_train[self.train_inds]
            self.y_train = self.y_train[self.train_inds]
            self.attr_train = self.attr_train[self.train_inds]
            if hasattr(self, 'y2_valid'):
                self.y2_valid = self.y2_train[self.valid_inds]
                self.y2_train = self.y2_train[self.train_inds]
        # hack for WGAN-GP training: trim to bacth size if a batch size is specified
        if self.batch_size is not None:
            self.x_train = mb_round(self.x_train, self.batch_size)
            self.x_test = mb_round(self.x_test, self.batch_size)
            self.x_valid = mb_round(self.x_valid, self.batch_size)
            self.attr_train = mb_round(self.attr_train, self.batch_size)
            self.attr_test = mb_round(self.attr_test, self.batch_size)
            self.attr_valid = mb_round(self.attr_valid, self.batch_size)
            self.y_train = mb_round(self.y_train, self.batch_size)
            self.y_test = mb_round(self.y_test, self.batch_size)
            self.y_valid = mb_round(self.y_valid, self.batch_size)
            if hasattr(self, 'y2_train'):
                self.y2_train = mb_round(self.y2_train, self.batch_size)
                self.y2_test = mb_round(self.y2_test, self.batch_size)
                self.y2_valid = mb_round(self.y2_valid, self.batch_size)

    def get_A_proportions(self):
        A0 = NR(self.attr_train)
        A1 = PR(self.attr_train)
        assert A0 + A1 == 1
        return [A0, A1]

    def get_Y_proportions(self):
        Y0 = NR(self.y_train)
        Y1 = PR(self.y_train)
        assert Y0 + Y1 == 1
        return [Y0, Y1]

    def get_AY_proportions(self):
        ttl = float(self.y_train.shape[0])
        A0Y0 = TN(self.y_train, self.attr_train) / ttl
        A0Y1 = FN(self.y_train, self.attr_train) / ttl
        A1Y0 = FP(self.y_train, self.attr_train) / ttl
        A1Y1 = TP(self.y_train, self.attr_train) / ttl
        return [[A0Y0, A0Y1], [A1Y0, A1Y1]]

    def get_batch_iterator(self, phase, mb_size):
        if phase == 'train':
            x = self.x_train
            y = self.y_train
            a = self.attr_train
        elif phase == 'valid':
            x = self.x_valid
            y = self.y_valid
            a = self.attr_valid
        elif phase == 'test':
            x = self.x_test
            y = self.y_test
            a = self.attr_test
        else:
            raise Exception("invalid phase name")

        sz = x.shape[0]
        batch_inds = make_batch_inds(sz, mb_size, self.seed, phase)
        iterator = DatasetIterator([x, y, a], batch_inds)
        return iterator


class TransferDataset(Dataset):
    def __init__(self, reprs, A, label_index, Y_loaded=None, phase='Test', **data_kwargs):
        super().__init__(**data_kwargs)
        if label_index == 'a':
            Y = A
        elif label_index >= 0:
            Y2 = self.y2_test if phase == 'Test' else self.y2_valid
            Y = np.expand_dims(Y2[:,label_index], 1)
        else:
            assert not Y_loaded is None
            Y = Y_loaded
            assert np.array_equal(Y, self.y_test) or np.array_equal(Y, self.y_valid)
        assert Y.shape[0] == reprs.shape[0]
        x_train, x_test, y_train, y_test, a_train, a_test = self.make_train_test_split(reprs, A, Y)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.attr_train = a_train
        self.attr_test = a_test
        self.train_inds, self.valid_inds = self.make_valid_inds(x_train, pct=0.2)
        self.make_validation_set(force=True)

    def make_valid_inds(self, X, pct):
        np.random.seed(self.seed)

        n = X.shape[0]
        shuf = np.arange(n)
        valid_pct = pct
        valid_ct = int(n * valid_pct)
        valid_inds = shuf[:valid_ct]
        train_inds = shuf[valid_ct:]

        return train_inds, valid_inds

    def make_train_test_split(self, X, A, Y):
        print(X.shape, A.shape, Y.shape)
        tr_inds, te_inds = self.make_valid_inds(X, pct=0.3)
        X_tr = X[tr_inds,:]
        X_te = X[te_inds,:]
        Y_tr = Y[tr_inds,:]
        Y_te = Y[te_inds,:]
        A_tr = A[tr_inds,:]
        A_te = A[te_inds,:]
        return X_tr, X_te, Y_tr, Y_te, A_tr, A_te


class DatasetIterator(collections.Iterator):
    def __init__(self, tensor_list, ind_list):
        self.tensors = tensor_list
        self.inds = ind_list
        self.curr = 0
        self.ttl_minibatches = len(self.inds)

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr >= self.ttl_minibatches:
            raise StopIteration
        else:
            inds = self.inds[self.curr]
            minibatch = [t[inds] for t in self.tensors]
            self.curr += 1
            return minibatch


def make_batch_inds(n, mb_size, seed=0, phase='train'):
    np.random.seed(seed)
    if phase == 'train':
        shuf = np.random.permutation(n)
    else:
        shuf = np.arange(n)
    start = 0
    mbs = []
    while start < n:
        end = min(start + mb_size, n)
        mb_i = shuf[start:end]
        mbs.append(mb_i)
        start = end
    return mbs
