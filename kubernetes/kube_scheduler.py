#!/usr/bin/env python3

from typing import Dict, Iterator
import dotenv
import hydra
from omegaconf import DictConfig

import os
import subprocess
from itertools import product

from jinja2 import Environment, FileSystemLoader

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

DATETIME_FORMAT = "%Y-%m-%d-%H-%M-%S"
TIME_FORMAT = "%H-%M-%S"
TEMPLATE_FILE = "kubernetes/kube.yaml.jinja2"


def script_args_to_string(dict_):
    string = ""
    args_list = [f"{k}={v}" for k, v in dict_.items()]
    string = " ".join(args_list)
    return string


def start_config(parameters, yaml, dry_run=False):
    template = Environment(loader=FileSystemLoader(hydra.utils.get_original_cwd())).get_template(TEMPLATE_FILE)
    print(hydra.utils.get_original_cwd())
    args_string = script_args_to_string(parameters)
    output_text = template.render(log_dir="/tmp", args_str=args_string, **yaml)
    if not dry_run:
        command = "kubectl -n dulny create -f -"
        p = subprocess.Popen(command.split(), stdin=subprocess.PIPE)
        p.communicate(output_text.encode())
    else:
        train_path = os.path.join(hydra.utils.get_original_cwd(), "train.py")
        command = f"python {train_path} debug=step "+args_string
        hydra_cwd = os.getcwd()
        original_cwd = hydra.utils.get_original_cwd()
        os.chdir(original_cwd)
        p = subprocess.Popen(command.split())
        p.communicate()
        os.chdir(hydra_cwd)


def get_hyperparam_iterator(hyperparams: dict) -> Iterator:
    hyperparam_iterator = []
    for values in product(*hyperparams.values()):
        keys = hyperparams.keys()
        dictionary = dict(zip(keys, values))
        hyperparam_iterator.append(dictionary)
    return hyperparam_iterator

@hydra.main(config_path="../configs/", config_name="kube_scheduler.yaml")
def main(config: DictConfig):
    # Arguments which will be passed to the python script. Boolean flags will be automatically set to "--key" (if True)
    yaml_dict = config.cluster

    hyperparameters = {
        #'model': ['resnet', 'pdernn', 'neuralPDE', 'cnn', 'convLSTM', 'distana', 'pdenet', 'persistence'],
        'model': ['neuralPDE'],
        'datamodule': ['gasdynamics', 'weatherbench', 'plasim', 'oceanwave']
    }

    def preprocess_hyperparams(p): # specific for this experiment
        input_dim_dict = {
            'gasdynamics': 4,
            'weatherbench': 2,
            'oceanwave': 3,
            'plasim': 4
        }
        image_size_dict = {
            'gasdynamics': 10*10,
            'weatherbench': 32*64,
            'oceanwave': 32*64,
            'plasim': 32*64,
        }
        p['model.net.input_dim'] = input_dim_dict[p['datamodule']] if p['model'] != 'pdernn' else input_dim_dict[p['datamodule']]*image_size_dict[p['datamodule']]
        p['name'] = p['model']
        if p['model'] == 'persistence':
            p['train'] = 'False'
        return p


    hyperparam_iterator = get_hyperparam_iterator(hyperparameters)
    
    for p in hyperparam_iterator:
        p = preprocess_hyperparams(p)
        start_config(p, yaml_dict, dry_run=True)


if __name__ == "__main__":
    main()
