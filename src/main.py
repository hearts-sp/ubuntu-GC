import numpy as np
import os
import collections
from os.path import dirname, abspath
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
import time
import random

from run import run
SETTINGS.CONFIG.READ_ONLY_CONFIG = False
SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log, env_args):
    # Setting the random seed throughout the modules
    random.seed(_config["seed"])
    np.random.seed(_config["seed"])
    th.manual_seed(_config["seed"])
    #th.cuda.seed(_config["seed"])
    #th.cuda.seed_all(_config["seed"])
    th.cuda.manual_seed(_config["seed"])
    th.cuda.manual_seed_all(_config["seed"])
    _config['env_args']['seed'] = _config["seed"]
    env_args['seed'] = _config["seed"]

    # run the framework
    run(_run, _config, _log)

    # force exit
    os._exit(0)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


if __name__ == '__main__':
    import os

    from copy import deepcopy
    params = deepcopy(sys.argv)
    train_map_name = "missing_map_name"
    eval_map_name = "missing_map_name"
    bool_eval = False
    for param in params:
        if "scenario" in param:
            if "eval" in param:
                eval_map_name = param.split("=")[1]
            else:
                train_map_name = param.split("=")[1]
        if "evaluate" in param:
            bool_eval = True if param.split("=")[1] == "True" else False

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    if bool_eval:
        env_config["random_tags"] = False
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    if True:  # only log if training
        # Save to disk by default for sacred
        logger.info("Saving to FileStorageObserver in results/sacred.")
        alg_name = config_dict['name']
        map_name = eval_map_name if bool_eval else train_map_name
        file_obs_path = os.path.join(results_path, "sacred/{}/{}".format(map_name, alg_name))
        while True:
            try:
                ex.observers.append(FileStorageObserver.create(file_obs_path))
                break
            except FileExistsError:
                # sometimes we see race condition
                logger.info("Creating FileStorageObserver failed. Trying again...")
                time.sleep(1)

    ex.run_commandline(params)

