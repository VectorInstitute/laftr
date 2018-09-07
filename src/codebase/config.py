'''
Boilerplate customizable JSON config loader.

Takes in a JSON config with overrides given as comma-separated options tiered with periods. Example:

python main.py config.json -o exp_name=test,data.name=cifar10

This code is due to James Lucas (@AtheMathmo) and Jake Snell (@jakesnell). Thanks!
'''
import argparse
import json
import os.path

import collections

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            try:
                d[k] = type(d[k])(v)
            except (TypeError, ValueError) as e:
                raise TypeError(e)  # types not compatible
            except KeyError as e:
                d[k] = v  # No matching key in dict

    return d


class ConfigParse(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        options_dict = {}
        for overrides in values.split(','):
            k, v = overrides.split('=')
            # process boolean options
            if v == 'True':
                v = True
            elif v == 'False':
                v = False
            k_parts = k.split('.')
            dic = options_dict
            for key in k_parts[:-1]:
                dic = dic.setdefault(key, {})
            dic[k_parts[-1]] = v
        setattr(namespace, self.dest, options_dict)


def get_config_overrides():
    parser = argparse.ArgumentParser(description='Experiments for fair representation learning')
    parser.add_argument('config', help='Base config file')
    parser.add_argument('-o', action=ConfigParse,
                        help='Config option overrides. Comma separated, e.g. optim.lr_init=1.0,optim.lr_decay=0.1')
    args, template_args = parser.parse_known_args()
    template_dict = dict(zip(template_args[:-1:2], template_args[1::2]))
    template_dict = {k.lstrip('-'): v for k, v in template_dict.items()}
    return args, template_dict


def process_config(verbose=True):
    args, template_args = get_config_overrides()

    with open(args.config, 'r') as f:
        template = f.read()

    env = Environment(loader=FileSystemLoader('conf/templates/'),
                      undefined=StrictUndefined)

    config = json.loads(env.from_string(template).render(**template_args))

    if args.o is not None:
        print(args.o)
        config = update(config, args.o)

    if verbose:
        print('-------- Config --------')
        print(json.dumps(config, indent=4, sort_keys=True))
        print('------------------------')
    return config


def save_config(conf, path):
    base_dir = os.path.dirname(path)

    if not os.path.isdir(base_dir):
        try:
            os.makedirs(base_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    with open(path, 'w') as f:
        json.dump(conf, f, sort_keys=True, indent=4)


def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

