def correct_repr_exists(opt):
    import os
    from codebase.config import load_config
    opt_filename = os.path.join(opt['dirs']['exp_dir'], opt['exp_name'], 'opt.json')
    done_filename_txt = os.path.join(opt['dirs']['exp_dir'], opt['exp_name'], 'done.txt')
    done_filename_json = os.path.join(opt['dirs']['exp_dir'], opt['exp_name'], 'done.json')
    print('opt_filename', opt_filename)
    print('done_filename_txt', done_filename_txt)
    print('done_filename_json', done_filename_json)
    if os.path.exists(done_filename_txt) or os.path.exists(done_filename_json) :
        if load_config(opt_filename) == opt:
            return True
    return False


if __name__ == '__main__':
    """
    This script takes LAFTR hyperparameters and trains a LAFTR model if a model with those hyperparameters values hasn't already been trained.
    But if it was previously trained (i.e. representations/encoder already exist) then we do nothing.
    It is meant to be used as part of a larger experiment sweep.

    See `src/laftr.py` for instructions on how to format arguments.
    """
    from codebase.config import process_config
    from laftr import main as learn_reps

    training_opt = process_config(verbose=False)
    training_opt.pop('transfer')  # can discard transfer params for purposes of learning the base rep

    if correct_repr_exists(training_opt):
        print('pre-trained reps exist, NOT TRAINING')
    else:
        print('learning reps from scratch')
        learn_reps(training_opt)

