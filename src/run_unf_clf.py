import copy
from run_laftr import correct_repr_exists
from transfer_learn import format_transfer_args_as_train_args
if __name__ == '__main__':
    """
    run from repo root
    first arg is base config file
    override individual params with -o then comma-separated (no spaces) list of args, e.g., -o exp_name=foo,data.seed=0,model.fair_coeff=2.0
    set templates by name, e.g., --data adult 
    
    e.g.,
    >>> python src/run_unf_clf.py conf/transfer/run_unf_clf.json -o train.n_epochs=5 --data adult --dirs local
    """
    from codebase.config import process_config
    from transfer_learn import get_repr_filename
    from transfer_learn import main as train_transfer_classifier
    from run_laftr import correct_repr_exists

    import os
    transfer_opt = process_config(verbose=False)
    training_opt = transfer_opt.copy()
    if transfer_opt['transfer']['repr_name'] == 'scratch':
        print('training classifier from data directly')
        train_transfer_classifier(transfer_opt)
    else:
        # this experiment gets its own reps and tranfsers
        repr_name, transfer_name = os.path.split(transfer_opt['exp_name'])
        assert 'transfer' in transfer_name, "expect a sweep over some transfer params"
        training_opt.update(exp_name=repr_name)
        transfer_opt['transfer'].update(repr_name=repr_name)
        training_opt.pop('transfer')  # can discard transfer params for purposes of learning the base rep
        assert correct_repr_exists(training_opt), "coudln't find pre-trained laftr representation :("
        maybe_done_json = '{}/{}/done.json'.format(transfer_opt['dirs']['exp_dir'], transfer_opt['exp_name'])
        if os.path.exists(maybe_done_json):
            print('classifier already trained, NOT TRAINING')
        else:
            print('training classifier')
            train_transfer_classifier(transfer_opt)
