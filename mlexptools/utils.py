import sys
from loguru import logger
from omegaconf import DictConfig, ListConfig

def get_logger(name: str, debug_flag: bool = False):
    logger.remove()
    logger.bind(name=name)
    logger.add(sys.stdout, level='DEBUG' if debug_flag else 'INFO')
    return logger


def omegaconf_to_dict(params):
    param_dict = {}
    for param_name, element in params.items():
        param_dict.update(_explore_recursive(param_name, element))
    return param_dict


def _explore_recursive(parent_name, element):
    params = {}
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if 'augmentation' in parent_name:
                continue
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                params.update(_explore_recursive(f'{parent_name}.{k}', v))
            else:
                params[f'{parent_name}.{k}'] = v
    else:
        params[parent_name] = element
    return params
