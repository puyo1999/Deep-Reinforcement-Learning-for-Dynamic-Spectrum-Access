from main import ROOT_DIR
import json
import numpy as np
import os
import pandas as pd
import pickle
import ujson
import yaml

def get_file_ext(data_path):
    '''get the `.ext` of file.ext'''
    return os.path.splitext(data_path)[-1]

def get_prepath(spec, unit='experiment'):
    spec_name = spec['name']
    #meta_spec = spec['meta']
    predir = f'file/{spec_name}'
    prename = f'{spec_name}'
    t_str = ''
    s_str = ''
    if unit == 'trial':
        prename += t_str
    elif unit == 'session':
        prename += f'{t_str}{s_str}'
    prepath = f'{predir}/{prename}'
    return prepath

def smart_path(data_path, as_dir=False):
    if not os.path.isabs(data_path):
        data_path = os.path.join(ROOT_DIR, data_path)

    if as_dir:
        data_path = os.path.dirname(data_path)
    return os.path.normpath(data_path)

def read_as_plain(data_path, **kwargs):
    open_file = open(data_path, 'r')
    ext = get_file_ext(data_path)
    if ext == '.json':
        data = ujson.load(open_file, **kwargs)
    elif ext == '.yml':
        data = yaml.load(open_file, **kwargs)
    else:
        data = open_file.read()
    open_file.close()
    return data

def read(data_path, **kwargs):
    '''
    :param data_path:
    :param kwargs:
    :return:
    '''
    data_path = smart_path(data_path)

    ext = get_file_ext(data_path)

    data = read_as_plain(data_path, **kwargs)
    return data

def write(data, data_path):
    '''
    Universal data writing method with smart data parsing
    - {.csv} from DataFrame
    - {.json} from dict, list
    - {.yml} from dict
    - {*} from str(*)
    @param {*} data The data to write
    @param {str} data_path The data path to write to
    @returns {data_path} The data path written to
    @example

    data_path = util.write(data_df, 'test/fixture/lib/util/test_df.csv')

    data_path = util.write(data_dict, 'test/fixture/lib/util/test_dict.json')
    data_path = util.write(data_dict, 'test/fixture/lib/util/test_dict.yml')

    data_path = util.write(data_list, 'test/fixture/lib/util/test_list.json')

    data_path = util.write(data_str, 'test/fixture/lib/util/test_str.txt')
    '''
    data_path = smart_path(data_path)
    data_dir = os.path.dirname(data_path)
    os.makedirs(data_dir, exist_ok=True)
    ext = get_file_ext(data_path)
    if ext == '.csv':
        write_as_df(data, data_path)
    elif ext == '.pkl':
        write_as_pickle(data, data_path)
    else:
        write_as_plain(data, data_path)
    return data_path

def cast_df(val):
    '''missing pydash method to cast value as DataFrame'''
    if isinstance(val, pd.DataFrame):
        return val
    return pd.DataFrame(val)


def write_as_df(data, data_path):
    '''Submethod to write data as DataFrame'''
    df = cast_df(data)
    df.to_csv(data_path, index=False)
    return data_path


def write_as_pickle(data, data_path):
    '''Submethod to write data as pickle'''
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    return data_path


def write_as_plain(data, data_path):
    '''Submethod to write data as plain type'''
    open_file = open(data_path, 'w')
    ext = get_file_ext(data_path)
    if ext == '.json':
        json.dump(data, open_file, indent=2, cls=LabJsonEncoder)
    elif ext == '.yml':
        yaml.dump(data, open_file)
    else:
        open_file.write(str(data))
    open_file.close()
    return data_path

class LabJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        else:
            return str(obj)