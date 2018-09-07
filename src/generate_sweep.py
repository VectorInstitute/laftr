from itertools import product
import os

from codebase.utils import make_comma_separated_args_dirname_friendly


def produce_lists(opt):
    overrides = opt['overrides']
    list_of_list_of_formatted_values = []
    for arg_type in sorted(overrides.keys()):
        for sub_arg_type in sorted(overrides[arg_type].keys()):
            list_of_list_of_formatted_values.append(["{}.{}={}".format(arg_type, sub_arg_type, value) for value in overrides[arg_type][sub_arg_type]])
    list_of_list_of_non_transfer_formatted_values = [l for l in list_of_list_of_formatted_values if not 'transfer' in l[0]]

    templates = opt['templates']
    static_template_args, swept_template_args = '', []
    for k, v in templates.items():
        if not isinstance(v, list):
            static_template_args += '--{} {} '.format(k, v)
        else:
            swept_template_args.append(['--{} {}'.format(k, vv) for vv in v])

    return list_of_list_of_formatted_values, \
               list_of_list_of_non_transfer_formatted_values, \
               static_template_args, \
               swept_template_args


def split_by_transfer(list_of_formatted_args):
    list_of_transfer_formatted_args = [a for a in list_of_formatted_args if 'transfer' in a]
    list_of_non_transfer_formatted_args = [a for a in list_of_formatted_args if not 'transfer' in a]
    return list_of_non_transfer_formatted_args, list_of_transfer_formatted_args


def main(opt, output_root):
    script_name = opt.pop('script')
    # if the scripts come in a list then the first trains representations with laftr and the second trains unfair classifiers
    script_names = script_name if isinstance(script_name, list) else [script_name]
    assert len(script_names) == 1 or len(script_names) == 2, "only one or two scripts"
    list_of_list_of_formatted_values, \
        list_of_list_of_non_transfer_formatted_values, \
        static_template_args, \
        swept_template_args = produce_lists(opt)
    sweep_name = opt.pop('sweep_name')


    COMMAND_FORMAT = '{script_name} {config_args} {template_args} -o {override_args},exp_name=\"{sweep_name}/{sweep_params_dir_friendly}\"'
    if opt['xargs']:
        commands_filenames = ['{}/arguments{}.txt'.format(output_root, i) for i in range(len(script_names))]
        srun_filename = '{}/go.sh'.format(output_root)
        # EXTRAS = '--gres=gpu:1 -p gpuc --pty'  # format srun args here
        #EXTRAS = "" if opt['nosrun'] else 'python -p gpuc --pty'  # format srun args here
        #EXTRAS = 'python -p gpuc --pty'  # format srun args here
        if 'partition' in opt:
            EXTRAS = '-p {partition} python'.format(**opt)  # format srun args here
        else:
            EXTRAS = '-p {} python'.format('cpu')  # format srun args here
        COMMAND_FORMAT = EXTRAS + " " + COMMAND_FORMAT  # add the extras up front
        # format the exclude guppies argument here
        if 'exclude_guppies' not in opt or opt['exclude_guppies'] is None:
            EXCLUDE_GUPPIES = ""
        else:
            EXCLUDE_GUPPIES  = "-x {exclude_guppies}".format(**opt)
        PROGRAM = "srun"
        if opt['n_proc'] is None:
            import numpy as np
            p = np.prod([len(x) for x in list_of_list_of_formatted_values + swept_template_args])
        else:
            p = opt['n_proc']
        srun_commands = ['xargs -n {n} -P {p} {program} {excludes} < {filename}'.format(
            n=len([cmd for cmd in COMMAND_FORMAT.split(" ") if not cmd in ["", "{template_args}"]]) + len(static_template_args.split(' ')[:-1]) + 2*len(swept_template_args),
            p=p,
            program=PROGRAM,
            excludes=EXCLUDE_GUPPIES,
            filename=commands_filename
        ) for commands_filename in commands_filenames]
        if os.path.exists(srun_filename):
            os.remove(srun_filename)
        if len(script_names) > 1:
            print(srun_commands[0], file=open(srun_filename, 'a'))
        print(srun_commands[-1], file=open(srun_filename, 'a'))
    else:
        commands_filename = '{}/commands.sh'.format(output_root)
    preamble = '' if opt['xargs'] else 'python '
    if len(script_names) > 1:  # first train rep via laftr then train unfair classifier
        list_of_commands_0 = [
            preamble + COMMAND_FORMAT.format(
                script_name=script_names[0],
                config_args=opt['config'],
                template_args=static_template_args + " ".join(v),
                override_args=",".join(u),
                sweep_name=sweep_name,
                sweep_params_dir_friendly=make_comma_separated_args_dirname_friendly(v, u, ())
            ) for u in product(*list_of_list_of_non_transfer_formatted_values) for v in product(*swept_template_args)
        ]
    list_of_commands = [
        preamble + COMMAND_FORMAT.format(
            script_name=script_names[-1],
            config_args=opt['config'],
            template_args=static_template_args + " ".join(v),
            override_args=",".join(u),
            sweep_name=sweep_name,
            sweep_params_dir_friendly=make_comma_separated_args_dirname_friendly(v, *split_by_transfer(u))
        ) for u in product(*list_of_list_of_formatted_values) for v in product(*swept_template_args)
    ]

    if opt['xargs']:
        c = [list_of_commands] if len(script_names) == 1 else [list_of_commands_0, list_of_commands]
        for commands_filename, commands in zip(commands_filenames, c):
            print("\n".join(commands), file=open(commands_filename, 'w'))
    else:
        if os.path.exists(commands_filename):
            os.remove(commands_filename)
        if len(script_names) > 1:
            print("\n".join(list_of_commands_0), file=open(commands_filename, 'a'))
        print("\n".join(list_of_commands), file=open(commands_filename, 'a'))
    if opt['xargs']:
        print('saved an srun command that automatically xargs arguments specified by the sweep config\n\tsrun command: \n\t\t{}'.format(srun_filename))
        print('\n\targuments')
        for commands_filename in commands_filenames:
            print('\n\t\t{}'.format(commands_filename))
    else:
        print('saved a list of commands that you can source from the terminal\n\tcommands:\n\t\t{}'.format(commands_filename))


if __name__ == '__main__':
    """
    This script generates formats a sweep experiment as a source-able bash file.
    It starts from a sweep config file (sweep.json) specifying sweepable lists of params and pointing to a base config file (config.json)

    Instructions:
    1) Run from repo root
    2) First arg is base sweep file
    3) All swept arguments should be specified the in sweep.json. 
    4) Default and non-swept parameters should be specified in the corresponding base config (config.json), which must be referred to in the sweep.json
    
    e.g.,
    >>> python src/generate_sweep.py sweeps/small_sweep_adult/sweep.json

    This command generates the sweep experiment as bash script at `sweeps/small_sweep_adult/commands.sh`.
    To run the experiment:
    >>> source sweep/small_sweep_adult/commands.sh

    The sweep generation supports some additional fanciness for running in parallel with slurm+xargs, which is not documented.
    """
    from codebase.config import process_config, get_config_overrides
    args, _ = get_config_overrides()
    sweep_file = args.config
    opt = process_config(verbose=False)
    main(opt, os.path.dirname(sweep_file))
