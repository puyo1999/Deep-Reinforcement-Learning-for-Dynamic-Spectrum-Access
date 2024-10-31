import json
import os
from string import Template
import itertools
from util import path_util
import pydash as ps
from py_lab.lib.logger import logger
from py_lab.lib import util

SPEC_DIR = 'config/spec'

SPEC_FORMAT = {
    "agent":[{
        "name": str,
        "algorithm": dict,
        "memory":dict,
        "net":dict,
    }],
    "env":[{
        "name": str,
        "num_hdmi": int,
        "num_dt": int,
        "num_mf": int,
        "num_cs": int,
        "attempt_prob": float,
        "observation_space": int
    }],
    "meta": {
        "max_session": int,
        "max_trial": (type(None), int),
    },
    "name": str,
}

def check_comp_spec(comp_spec, comp_spec_format):
    '''Base method to check component spec'''
    for spec_k, spec_format_v in comp_spec_format.items():
        logger.error(f'spec_k:{spec_k}\n spec_format_v:{spec_format_v}')
        comp_spec_v = comp_spec[spec_k]
        logger.error(f'comp_spec_v:{comp_spec_v}')
        if ps.is_list(spec_format_v):
            v_set = spec_format_v
            assert comp_spec_v in v_set, f'Component spec value {ps.pick(comp_spec, spec_k)} needs to be one of {util.to_json(v_set)}'
        else:
            v_type = spec_format_v
            assert isinstance(comp_spec_v, v_type), f'Component spec {ps.pick(comp_spec, spec_k)} needs to be of type: {v_type}'
            if isinstance(v_type, tuple) and int in v_type and isinstance(comp_spec_v, float):
                # cast if it can be int
                comp_spec[spec_k] = int(comp_spec_v)

def check_compatibility(spec):
    '''Check compatibility among spec setups'''
    # TODO expand to be more comprehensive
    if spec['meta'].get('distributed') == 'synced':
        assert ps.get(spec, 'agent.0.net.gpu') == False, f'Distributed mode "synced" works with CPU only. Set gpu: false.'

def check(spec):
    '''Check a single spec for validity'''
    try:
        spec_name = spec.get('name')
        logger.info(f'spec_name : {spec_name}')
        assert set(spec.keys()) >= set(SPEC_FORMAT.keys()), f'Spec needs to follow spec.SPEC_FORMAT. Given \n {spec_name}: {util.to_json(spec)}'
        for agent_spec in spec['agent']:
            check_comp_spec(agent_spec, SPEC_FORMAT['agent'][0])
        for env_spec in spec['env']:
            check_comp_spec(env_spec, SPEC_FORMAT['env'][0])
        #check_comp_spec(spec['body'], SPEC_FORMAT['body'])
        check_comp_spec(spec['meta'], SPEC_FORMAT['meta'])
        # check_body_spec(spec)
        check_compatibility(spec)
    except Exception as e:
        logger.exception(f'spec {spec_name} fails spec check')
        raise e
    return True



def get(spec_file, spec_name, exp_ts=None):
    '''
    Get an experiment spec from spec_file, spec_name.
    Auto-check spec.
    @param str:spec_file
    @param str:spec_name
    @param str:experiment_ts Use this experiment_ts if given; used for resuming training
    @example

    spec = spec_util.get('demo.json', 'ppo_mbr_24_09_17')
    '''
    spec_file = spec_file.replace(SPEC_DIR, '')
    spec_file = f'{SPEC_DIR}/{spec_file}'
    spec_dict = path_util.read(spec_file)

    assert spec_name in spec_dict, f'spec_name {spec_name} is not in spec_file {spec_file}. Choose from:\n {ps.join(spec_dict.keys(), ",")}'

    print(f'spec_file:{spec_file}')
    print(f'spec_dict:{spec_dict}')

    spec = spec_dict[spec_name]
    print(f'@@@ spec :\n{spec}')

    spec['name'] = spec_name
    print(f'@@@ spec[name] :\n {spec["name"]}')
    # fill-in info at runtime
    spec = extend_meta_spec(spec, exp_ts)
    print(f'@@@ agent :\n {spec["agent"]}')
    check(spec)
    return spec
def extend_meta_spec(spec, exp_ts):
    '''
    :param spec:
    :param exp_ts:
    :return: spec
    '''
    extended_meta_spec = {
        'rigorous_eval': ps.get(spec, 'meta.rigorous_eval', 0),
        # reset lab indices to -1 so that they tick to 0
        'experiment': -1,
        'trial': -1,
        'session': -1,
        'cuda_offset': int(os.environ.get('CUDA_OFFSET', 0)),
        'resume': exp_ts is not None,
        'experiment_ts': exp_ts or util.get_ts(),
        'prepath': None,
        'git_sha': util.get_git_sha(),
        'random_seed': None,
    }
    spec['meta'].update(extended_meta_spec)
    return spec
def get_param_specs(spec):
    '''Return a list of specs with substituted spec_params'''
    assert 'spec_params' in spec, 'Parametrized spec needs a spec_params key'
    spec_params = spec.pop('spec_params')
    spec_template = Template(json.dumps(spec))
    keys = spec_params.keys()
    specs = []
    for idx, vals in enumerate(itertools.product(*spec_params.values())):
        spec_str = spec_template.substitute(dict(zip(keys, vals)))
        spec = json.loads(spec_str)
        spec['name'] += f'_{"_".join(vals)}'
        # offset to prevent parallel-run GPU competition, to mod in util.set_cuda_id
        #spec['meta']['cuda_offset'] += idx * spec['meta']['max_session']
        specs.append(spec)
    return specs

def _override_dev_spec(spec):
    #spec['meta']['max_session'] = 1
    #spec['meta']['max_trial'] = 2
    return spec

def _override_enjoy_spec(spec):
    #spec['meta']['max_session'] = 1
    return spec

def override_spec(spec, mode):
    overrider = {
        'dev': _override_dev_spec,
        'enjoy': _override_enjoy_spec,
        #'test': _override_test_spec,
    }.get(mode)
    if overrider is not None:
        overrider(spec)
    return spec

def save(spec, unit='experiment'):
    prepath = path_util.get_prepath(spec, unit)
    path_util.write(spec, f'{prepath}_spec.json')

def reload(spec, unit='experiment'):
    prepath = path_util.get_prepath(spec, unit)
    path_util.read(prepath)

