from json import dumps
import numpy as np
import os
import tensorflow as tf

from codebase.utils import make_dir_if_not_exist
from codebase.datasets import TransferDataset, Dataset
from codebase.models import RegularizedFairClassifier
from codebase.trainer import Trainer
from codebase.tester import Tester
from codebase.results import ResultLogger
from codebase.utils import get_npz_basename


def format_transfer_args_as_train_args(opt):
    """replace opt['train'] with opt['transfer'] and update values as appropriate
    the point of this is that the -finetune/opt.json should be interpretable as params for a classifier training
    however, you won't be able to rerun transfer_learn by pointing to this opt. use the non-finetine opt instead"""
    def _get_repr_dim(args):
        npzdname, _ = get_repr_filename(args)
        repr_fname = os.path.join(npzdname, 'Z.npz')
        repr_dat = np.load(repr_fname)
        reprs = repr_dat['X']
        return reprs.shape[1]
    # use transfer-specified training params for training
    opt = opt.copy()
    for a in ['batch_size', 'n_epochs', 'patience']:
        opt['train'].update({a: opt['transfer'].pop(a)})
    opt['train'].update(regbas=True)  # in the transfer setup we always want to train the feedforward MLP
    # update the model to be a regularized fair classifier
    opt['model'].update({'class': 'RegularizedFairClassifier'})
    if opt['transfer']['repr_name'] != 'default':
        repr_dim = _get_repr_dim(opt)
        opt['model'].update(xdim=repr_dim)
        opt['model'].update(zdim=repr_dim//2)  # note: the choice of z-dim = x-dim / 2 is a ad-hoc
    else:
        opt['model'].update(xdim=opt['model']['xdim'] + int(opt['data']['use_attr']))  # for use_attr == True
        opt['model'].update(zdim=opt['model']['zdim'])
    opt['model'].update(hidden_layer_specs=dict(enc=[], cla=[], rec=[], aud=[])) #opt['model']['zdim']
    opt['model'].update(recon_coeff=0., fair_coeff=0., class_coeff=1.)  # vanilla unfair classifier
    if 'model_seed' in opt['transfer']:
        opt['model'].update(seed=opt['transfer'].pop('model_seed'))  # permit different initial weights at transfer time
    return opt


def get_repr_filename(args):
    if args['transfer']['epoch_number'] is not None:
        if not args['transfer']['epoch_number'] in ['DI', 'DP', 'DI_FP', 'ErrA']:
            repoch = int(args['transfer']['epoch_number'])
        else:
            #get (predetermined) best epoch
            bep_file = os.path.join(args['dirs']['repr_dir'], args['transfer']['repr_name'], 'best_validation_fairness.txt')
            bep_info = {l.split(',')[0]: l.strip().split(',')[1] for l in open(bep_file, 'r').readlines()}
            repoch = int(bep_info[args['transfer']['epoch_number']])
            print(bep_file)
        npzdname = os.path.join(args['dirs']['repr_dir'], args['transfer']['repr_name'], 'checkpoints',\
                'Epoch_{:d}_{}'.format(repoch, args['transfer']['repr_phase']), 'npz')
        print(npzdname)
    else:
        npzdname = os.path.join(args['dirs']['repr_dir'], args['transfer']['repr_name'], 'npz')
    opt_filename = os.path.join(args['dirs']['repr_dir'], args['transfer']['repr_name'], 'opt.json')
    return npzdname, opt_filename


def main(args):
    args = format_transfer_args_as_train_args(args)
    # things with paths
    expdname = args['dirs']['exp_dir']
    expname = args['exp_name']
    logdname = args['dirs']['log_dir']
    resdirname = os.path.join(expdname, expname)
    make_dir_if_not_exist(resdirname)
    logdirname = os.path.join(logdname, expname)
    make_dir_if_not_exist(logdirname)
    npzfile = os.path.join(args['dirs']['data_dir'],
                           args['data']['name'],
                           get_npz_basename(**args['data']))

    # write params (including all overrides) to experiment directory
    with open(os.path.join(resdirname, 'done.txt'), 'w') as f:
        f.write('done')

    npzdname, _ = get_repr_filename(args)
    repr_phase = args['transfer']['repr_phase']  # this will be "Test" or "Valid"
    y_indices = args['transfer']['y_indices'] if hasattr(args['transfer']['y_indices'], '__iter__') else [args['transfer']['y_indices']]

    if args['transfer']['repr_name'] == 'default':
        base_data = Dataset(npzfile=npzfile, **args['data'], batch_size=args['train']['batch_size'])
        if repr_phase == 'Test':
            reprs = base_data.x_test
            attrs = base_data.attr_test
            y = base_data.y_test
        elif repr_phase == 'Valid':
            reprs = base_data.x_valid
            attrs = base_data.attr_valid
            y = base_data.y_valid
    else:
        # load reprs from LAFTR training as input
        repr_fname = os.path.join(npzdname, 'Z.npz')
        repr_dat = np.load(repr_fname)
        reprs = repr_dat['X']

        attr_fname = os.path.join(npzdname, 'A.npz')
        attr_dat = np.load(attr_fname)
        attrs = attr_dat['X']

        y_fname = os.path.join(npzdname, 'Y.npz')
        y_dat = np.load(y_fname)
        y = y_dat['X']
    print('shapes', reprs.shape, attrs.shape, y.shape)

    for label_index in y_indices:  # this will either be a list of ints or just 'a'
        data = TransferDataset(reprs, attrs, label_index, npzfile=npzfile, Y_loaded=y, phase=repr_phase, **args['data'], batch_size=args['train']['batch_size'])
        model = RegularizedFairClassifier(**args['model'])

        with tf.Session() as sess:
            reslogger = ResultLogger(resdirname)

            # create Trainer
            trainer = Trainer(model, data, sess=sess, expdir=resdirname, logs_path=logdirname,
                              **args['optim'], **args['train'])
            trainer.train(**args['train']) #, train_metric=train_metric)

            # test model
            tester = Tester(model, data, sess, reslogger)
            tester.evaluate(args['train']['batch_size'])

        # flush
        tf.reset_default_graph()

    # all done
    with open(os.path.join(resdirname, 'done.json'), 'w') as f:
        opt_dumps = dumps(args, indent=4, sort_keys=True)
        f.write(opt_dumps)


if __name__ == '__main__':
    """
     for use by `src/run_unf_clf.py` in the context of a sweep.
     """
    from codebase.config import process_config
    opt = process_config(verbose=False)
    main(opt)



