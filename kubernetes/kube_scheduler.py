#!/usr/bin/env python3

from typing import Dict, Iterator
import dotenv
import hydra
from omegaconf import DictConfig
from time import sleep

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


def start_config(base, script, yaml, dry_run=False):
    script_path = f"scripts/{base}/{script}.sh"
    yaml['job_name'] = f"neuralpde-{script}"
    template = Environment(loader=FileSystemLoader(hydra.utils.get_original_cwd())).get_template(TEMPLATE_FILE)
    output_text = template.render(log_dir="/tmp", args_str=script_path, **yaml)
    if not dry_run:
        command = "kubectl -n dulny create -f -"
        p = subprocess.Popen(command.split(), stdin=subprocess.PIPE)
        p.communicate(output_text.encode())
        sleep(0.5)


@hydra.main(config_path="../configs/", config_name="kube_scheduler.yaml")
def main(config: DictConfig):
    # Arguments which will be passed to the python script. Boolean flags will be automatically set to "--key" (if True)
    yaml_dict = config.cluster

    scripts = []
    script_dict = {
        "neuralpde": ["neuralpdeadvection", "neuralpdewave","neuralpdeburgers","neuralpdegasdynamics","neuralpdeoceanwave","neuralpdeweatherbench", "neuralpdeplasim"],
        "neuralpde2": ["neuralpde2advection", "neuralpde2wave","neuralpde2burgers","neuralpde2gasdynamics","neuralpde2oceanwave","neuralpde2weatherbench", "neuralpde2plasim"],
        "hiddenstate": ["hiddenstateadvection", "hiddenstatewave","hiddenstateburgers","hiddenstategasdynamics","hiddenstateoceanwave","hiddenstateweatherbench", "hiddenstateplasim"]
    }
    script_dict = {
        "neuralpde2": ["neuralpde2plasim"],
    }
    
    for base, scripts in script_dict.items():
        for script in scripts:
            start_config(base, script, yaml_dict, dry_run=False)


if __name__ == "__main__":
    main()
