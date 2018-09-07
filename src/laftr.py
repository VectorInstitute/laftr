from json import dumps
import sys
import os
import tensorflow as tf
from codebase.datasets import Dataset
from codebase import models
from codebase.trainer import Trainer
from codebase.tester import Tester
from codebase.results import ResultLogger
from codebase.utils import get_npz_basename, make_dir_if_not_exist

def main(args):
    # things with paths
    expdname = args['dirs']['exp_dir']
    expname = args['exp_name']
    logdname = args['dirs']['log_dir']
    resdirname = os.path.join(expdname, expname)
    #logdirname = os.path.join(logdname, expname)
    logdirname = resdirname
    make_dir_if_not_exist(logdirname, remove=True)
    make_dir_if_not_exist(resdirname)
    npzfile = os.path.join(args['dirs']['data_dir'],
                           args['data']['name'],
                           get_npz_basename(**args['data']))

    # write params (including all overrides) to experiment directory
    with open(os.path.join(resdirname, 'opt.json'), 'w') as f:
        opt_dumps = dumps(args, indent=4, sort_keys=True)
        f.write(opt_dumps)

    if args['data']['use_attr']:
        args['model'].update(xdim=args['model']['xdim']+1)


    #get dataset
    data = Dataset(npzfile=npzfile, **args['data'], batch_size=args['train']['batch_size'])

    # get model
    if 'Weighted' in args['model']['class']:
        A_weights = [1. / x for x in data.get_A_proportions()]
        Y_weights = [1. / x for x in data.get_Y_proportions()]
        AY_weights = [[1. / x for x in L] for L in data.get_AY_proportions()]
        if 'Eqopp' in args['model']['class']:
            #we only care about ppl with Y = 0 --- those who didn't get sick
            AY_weights[0][1] = 0. #AY_weights[0][1]
            AY_weights[1][1] = 0. #AY_weights[1][1]
        args['model'].update(A_weights=A_weights, Y_weights=Y_weights, AY_weights=AY_weights)
    model_class = getattr(models, args['model'].pop('class'))
    model = model_class(**args['model'], batch_size=args['train']['batch_size'])

    with tf.Session() as sess:
        reslogger = ResultLogger(resdirname)

        #create Trainer
        trainer = Trainer(model, data, sess=sess, expdir=resdirname, logs_path=logdirname,
                          **args['optim'], **args['train'])

        # training
        trainer.train(**args['train'])

        # test the trained model
        tester = Tester(model, data, sess, reslogger)
        tester.evaluate(args['train']['batch_size'])

    # flush
    tf.reset_default_graph()

    # all done
    with open(os.path.join(resdirname, 'done.txt'), 'w') as f:
        f.write('done')


if __name__ == '__main__':
    """
    This script trains a LAFTR model. For the full evaluation used in the paper (first train LAFTR then evaluate on naive classifier) see `src/run_laftr.py`.

    Instructions: 
    1) Run from repo root
    2) First arg is base config file
    3) Optionally override individual params with -o then comma-separated (no spaces) list of args, e.g., -o exp_name=foo,data.seed=0,model.fair_coeff=2.0 
       (the overrides must come before the named templates in steps 4--5)
    4) Set templates by name, e.g., --data adult 
    5) Required templates are --data and --dirs
    
    e.g.,
    >>> python src/laftr.py conf/laftr/config.json -o train.n_epochs=10,model.fair_coeff=2. --data adult --dirs local

    This command trains LAFTR on the Adult dataset for ten epochs with batch size 32.
    Model and optimization parameters are specified by the config file conf/classification/config.json
    Dataset specifications are read from conf/templates/data/adult.json
    Directory specifications are read from conf/templates/dirs/local.json
    Finally, two hyperparameters are overridden by the last command.
    By using the -o flag we train for 10 epochs with fairness regulariztion coeff 2. instead of the default values from the config.json.
    """
    from codebase.config import process_config
    opt = process_config(verbose=False)
    main(opt)

