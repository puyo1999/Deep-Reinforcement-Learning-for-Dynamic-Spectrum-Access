from collections import deque
from contextlib import contextmanager
from datetime import datetime
from pprint import pformat
from main import ROOT_DIR, EVAL_MODES, TRAIN_MODES
import importlib
import cv2
import json
import numpy as np
import operator
import os
import pandas as pd
import pickle
import pydash as ps
import regex as re
import subprocess
import sys
import time
import torch
import torch.multiprocessing as mp
import ujson
import yaml

NUM_CPUS = mp.cpu_count()
FILE_TS_FORMAT = '%Y_%m_%d_%H%M%S'
RE_FILE_TS = re.compile(r'(\d{4}_\d{2}_\d{2}_\d{6})')


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


def batch_get(arr, idxs):
    '''Get multi-idxs from an array depending if it's a python list or np.array'''
    if isinstance(arr, (list, deque)):
        return np.array(operator.itemgetter(*idxs)(arr))
    else:
        return arr[idxs]


def calc_srs_mean_std(sr_list):
    '''Given a list of series, calculate their mean and std'''
    cat_df = pd.DataFrame(dict(enumerate(sr_list)))
    mean_sr = cat_df.mean(axis=1)
    std_sr = cat_df.std(axis=1)
    return mean_sr, std_sr


def calc_ts_diff(ts2, ts1):
    '''
    Calculate the time from tss ts1 to ts2
    @param {str} ts2 Later ts in the FILE_TS_FORMAT
    @param {str} ts1 Earlier ts in the FILE_TS_FORMAT
    @returns {str} delta_t in %H:%M:%S format
    @example

    ts1 = '2017_10_17_084739'
    ts2 = '2017_10_17_084740'
    ts_diff = util.calc_ts_diff(ts2, ts1)
    # => '0:00:01'
    '''
    delta_t = datetime.strptime(ts2, FILE_TS_FORMAT) - datetime.strptime(ts1, FILE_TS_FORMAT)
    return str(delta_t)


def cast_df(val):
    '''missing pydash method to cast value as DataFrame'''
    if isinstance(val, pd.DataFrame):
        return val
    return pd.DataFrame(val)


def cast_list(val):
    '''missing pydash method to cast value as list'''
    if ps.is_list(val):
        return val
    else:
        return [val]


def concat_batches(batches):
    '''
    Concat batch objects from body.memory.sample() into one batch, when all bodies experience similar envs
    Also concat any nested epi sub-batches into flat batch
    {k: arr1} + {k: arr2} = {k: arr1 + arr2}
    '''
    # if is nested, then is episodic
    is_episodic = isinstance(batches[0]['dones'][0], (list, np.ndarray))
    concat_batch = {}
    for k in batches[0]:
        datas = []
        for batch in batches:
            data = batch[k]
            if is_episodic:  # make into plain batch instead of nested
                data = np.concatenate(data)
            datas.append(data)
        concat_batch[k] = np.concatenate(datas)
    return concat_batch


def downcast_float32(df):
    '''Downcast any float64 col to float32 to allow safer pandas comparison'''
    for col in df.columns:
        if df[col].dtype == 'float':
            df[col] = df[col].astype('float32')
    return df


def epi_done(done):
    '''
    General method to check if episode is done for both single and vectorized env
    Only return True for singleton done since vectorized env does not have a natural episode boundary
    '''
    return np.isscalar(done) and done


def frame_mod(frame, frequency, num_envs):
    '''
    Generic mod for (frame % frequency == 0) for when num_envs is 1 or more,
    since frame will increase multiple ticks for vector env, use the remainder'''
    remainder = num_envs or 1
    return (frame % frequency < remainder)


def flatten_dict(obj, delim='.'):
    '''Missing pydash method to flatten dict'''
    nobj = {}
    for key, val in obj.items():
        if ps.is_dict(val) and not ps.is_empty(val):
            strip = flatten_dict(val, delim)
            for k, v in strip.items():
                nobj[key + delim + k] = v
        elif ps.is_list(val) and not ps.is_empty(val) and ps.is_dict(val[0]):
            for idx, v in enumerate(val):
                nobj[key + delim + str(idx)] = v
                if ps.is_object(v):
                    nobj = flatten_dict(nobj, delim)
        else:
            nobj[key] = val
    return nobj


def get_class_name(obj, lower=False):
    '''Get the class name of an object'''
    class_name = obj.__class__.__name__
    if lower:
        class_name = class_name.lower()
    return class_name


def get_class_attr(obj):
    '''Get the class attr of an object as dict'''
    attr_dict = {}
    for k, v in obj.__dict__.items():
        if hasattr(v, '__dict__') or ps.is_tuple(v):
            val = str(v)
        else:
            val = v
        attr_dict[k] = val
    return attr_dict


def get_file_ext(data_path):
    '''get the `.ext` of file.ext'''
    return os.path.splitext(data_path)[-1]


def get_fn_list(a_cls):
    '''
    Get the callable, non-private functions of a class
    @returns {[*str]} A list of strings of fn names
    '''
    fn_list = ps.filter_(dir(a_cls), lambda fn: not fn.endswith('__') and callable(getattr(a_cls, fn)))
    return fn_list


def get_git_sha():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], close_fds=True, cwd=ROOT_DIR).decode().strip()


def get_lab_mode():
    return os.environ.get('lab_mode')


def get_port():
    '''Get a unique port number for a run time as 4xxx, where xxx is the last 3 digits from the PID, front-padded with 0'''
    # get 3 digits from pid
    xxx = ps.pad_start(str(os.getpid())[-3:], 3, 0)
    port = int(f'4{xxx}')
    return port


def get_prepath(spec, unit='experiment'):
    spec_name = spec['name']
    meta_spec = spec['meta']
    predir = f'data/{spec_name}_{meta_spec["experiment_ts"]}'
    prename = f'{spec_name}'
    trial_index = meta_spec['trial']
    session_index = meta_spec['session']
    t_str = '' if trial_index is None else f'_t{trial_index}'
    s_str = '' if session_index is None else f'_s{session_index}'
    if unit == 'trial':
        prename += t_str
    elif unit == 'session':
        prename += f'{t_str}{s_str}'
    prepath = f'{predir}/{prename}'
    return prepath


def get_session_df_path(session_spec, df_mode):
    '''Method to return standard filepath for session_df (agent.body.train_df/eval_df) for saving and loading'''
    info_prepath = session_spec['meta']['info_prepath']
    return f'{info_prepath}_session_df_{df_mode}.csv'


def get_ts(pattern=FILE_TS_FORMAT):
    '''
    Get current ts, defaults to format used for filename
    @param {str} pattern To format the ts
    @returns {str} ts
    @example

    util.get_ts()
    # => '2017_10_17_084739'
    '''
    ts_obj = datetime.now()
    ts = ts_obj.strftime(pattern)
    assert RE_FILE_TS.search(ts)
    return ts


def insert_folder(prepath, folder):
    '''Insert a folder into prepath'''
    split_path = prepath.split('/')
    prename = split_path.pop()
    split_path += [folder, prename]
    return '/'.join(split_path)


def in_eval_lab_mode():
    '''Check if lab_mode is one of EVAL_MODES'''
    return get_lab_mode() in EVAL_MODES


def in_train_lab_mode():
    '''Check if lab_mode is one of TRAIN_MODES'''
    return get_lab_mode() in TRAIN_MODES


def is_jupyter():
    '''Check if process is in Jupyter kernel'''
    try:
        get_ipython().config
        return True
    except NameError:
        return False
    return False


@contextmanager
def ctx_lab_mode(lab_mode):
    '''
    Creates context to run method with a specific lab_mode
    @example
    with util.ctx_lab_mode('eval'):
        foo()

    @util.ctx_lab_mode('eval')
    def foo():
        ...
    '''
    prev_lab_mode = os.environ.get('lab_mode')
    os.environ['lab_mode'] = lab_mode
    yield
    if prev_lab_mode is None:
        del os.environ['lab_mode']
    else:
        os.environ['lab_mode'] = prev_lab_mode


def monkey_patch(base_cls, extend_cls):
    '''Monkey patch a base class with methods from extend_cls'''
    ext_fn_list = get_fn_list(extend_cls)
    for fn in ext_fn_list:
        setattr(base_cls, fn, getattr(extend_cls, fn))


def parallelize(fn, args, num_cpus=NUM_CPUS):
    '''
    Parallelize a method fn, args and return results with order preserved per args.
    args should be a list of tuples.
    @returns {list} results Order preserved output from fn.
    '''
    pool = mp.Pool(num_cpus, maxtasksperchild=1)
    results = pool.starmap(fn, args)
    pool.close()
    pool.join()
    return results


def prepath_split(prepath):
    '''
    Split prepath into useful names. Works with predir (prename will be None)
    prepath: data/dqn_pong_2018_12_02_082510/dqn_pong_t0_s0
    predir: data/dqn_pong_2018_12_02_082510
    prefolder: dqn_pong_2018_12_02_082510
    prename: dqn_pong_t0_s0
    spec_name: dqn_pong
    experiment_ts: 2018_12_02_082510
    '''
    prepath = prepath.strip('_')
    tail = prepath.split('data/')[-1]
    if '/' in tail:  # tail = prefolder/prename
        prefolder, prename = tail.split('/', 1)
    else:
        prefolder, prename = tail, None
    predir = f'data/{prefolder}'
    spec_name = RE_FILE_TS.sub('', prefolder).strip('_')
    experiment_ts = RE_FILE_TS.findall(prefolder)[0]
    return predir, prefolder, prename, spec_name, experiment_ts


def prepath_to_idxs(prepath):
    '''Extract trial index and session index from prepath if available'''
    tidxs = re.findall(r'_t(\d+)', prepath)
    trial_index = int(tidxs[0]) if tidxs else None
    sidxs = re.findall(r'_s(\d+)', prepath)
    session_index = int(sidxs[0]) if sidxs else None
    return trial_index, session_index


def read(data_path, **kwargs):
    '''
    Universal data reading method with smart data parsing
    - {.csv} to DataFrame
    - {.json} to dict, list
    - {.yml} to dict
    - {*} to str
    @param {str} data_path The data path to read from
    @returns {data} The read data in sensible format
    @example

    data_df = util.read('test/fixture/lib/util/test_df.csv')
    # => <DataFrame>

    data_dict = util.read('test/fixture/lib/util/test_dict.json')
    data_dict = util.read('test/fixture/lib/util/test_dict.yml')
    # => <dict>

    data_list = util.read('test/fixture/lib/util/test_list.json')
    # => <list>

    data_str = util.read('test/fixture/lib/util/test_str.txt')
    # => <str>
    '''
    data_path = smart_path(data_path)
    try:
        assert os.path.isfile(data_path)
    except AssertionError:
        raise FileNotFoundError(data_path)
    ext = get_file_ext(data_path)
    if ext == '.csv':
        data = read_as_df(data_path, **kwargs)
    elif ext == '.pkl':
        data = read_as_pickle(data_path, **kwargs)
    else:
        data = read_as_plain(data_path, **kwargs)
    return data


def read_as_df(data_path, **kwargs):
    '''Submethod to read data as DataFrame'''
    data = pd.read_csv(data_path, **kwargs)
    return data


def read_as_pickle(data_path, **kwargs):
    '''Submethod to read data as pickle'''
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def read_as_plain(data_path, **kwargs):
    '''Submethod to read data as plain type'''
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


def self_desc(cls, omit=None):
    '''Method to get self description, used at init.'''
    desc_list = [f'{get_class_name(cls)}:']
    omit_list = ps.compact(cast_list(omit))
    for k, v in get_class_attr(cls).items():
        if k in omit_list:
            continue
        if k == 'spec':  # spec components are described at their object level; for session, only desc spec.meta
            desc_v = pformat(v['meta'])
        elif ps.is_dict(v) or ps.is_dict(ps.head(v)):
            desc_v = pformat(v)
        else:
            desc_v = v
        desc_list.append(f'- {k} = {desc_v}')
    desc = '\n'.join(desc_list)
    return desc


def set_attr(obj, attr_dict, keys=None):
    '''Set attribute of an object from a dict'''
    if keys is not None:
        attr_dict = ps.pick(attr_dict, keys)
    for attr, val in attr_dict.items():
        setattr(obj, attr, val)
    return obj


def set_cuda_id(spec):
    '''Use trial and session id to hash and modulo cuda device count for a cuda_id to maximize device usage. Sets the net_spec for the base Net class to pick up.'''
    # Don't trigger any cuda call if not using GPU. Otherwise will break multiprocessing on machines with CUDA.
    # see issues https://github.com/pytorch/pytorch/issues/334 https://github.com/pytorch/pytorch/issues/3491 https://github.com/pytorch/pytorch/issues/9996
    for agent_spec in spec['agent']:
        if not agent_spec['net'].get('gpu'):
            return
    meta_spec = spec['meta']
    trial_idx = meta_spec['trial'] or 0
    session_idx = meta_spec['session'] or 0
    if meta_spec['distributed'] == 'shared':  # shared hogwild uses only global networks, offset them to idx 0
        session_idx = 0
    job_idx = trial_idx * meta_spec['max_session'] + session_idx
    job_idx += meta_spec['cuda_offset']
    device_count = torch.cuda.device_count()
    cuda_id = job_idx % device_count if torch.cuda.is_available() else None

    for agent_spec in spec['agent']:
        agent_spec['net']['cuda_id'] = cuda_id


def set_logger(spec, logger, unit=None):
    '''Set the logger for a lab unit give its spec'''
    os.environ['LOG_PREPATH'] = insert_folder(get_prepath(spec, unit=unit), 'log')
    importlib.reload(logger)  # to set session-specific logger


def set_random_seed(spec):
    '''Generate and set random seed for relevant modules, and record it in spec.meta.random_seed'''
    trial = spec['meta']['trial']
    session = spec['meta']['session']
    random_seed = int(1e5 * (trial or 0) + 1e3 * (session or 0) + time.time())
    torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    spec['meta']['random_seed'] = random_seed
    return random_seed


def _sizeof(obj, seen=None):
    '''Recursively finds size of objects'''
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([_sizeof(v, seen) for v in obj.values()])
        size += sum([_sizeof(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += _sizeof(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([_sizeof(i, seen) for i in obj])
    return size


def sizeof(obj, divisor=1e6):
    '''Return the size of object, in MB by default'''
    return _sizeof(obj) / divisor


def smart_path(data_path, as_dir=False):
    '''
    Resolve data_path into abspath with fallback to join from ROOT_DIR
    @param {str} data_path The input data path to resolve
    @param {bool} as_dir Whether to return as dirname
    @returns {str} The normalized absolute data_path
    @example

    util.smart_path('slm_lab/lib')
    # => '/Users/ANON/Documents/slm_lab/slm_lab/lib'

    util.smart_path('/tmp')
    # => '/tmp'
    '''
    if not os.path.isabs(data_path):
        data_path = os.path.join(ROOT_DIR, data_path)
    if as_dir:
        data_path = os.path.dirname(data_path)
    return os.path.normpath(data_path)


def split_minibatch(batch, mb_size):
    '''Split a batch into minibatches of mb_size or smaller, without replacement'''
    size = len(batch['rewards'])
    assert mb_size < size, f'Minibatch size {mb_size} must be < batch size {size}'
    idxs = np.arange(size)
    np.random.shuffle(idxs)
    chunks = int(size / mb_size)
    nested_idxs = np.array_split(idxs[:chunks * mb_size], chunks)
    if size % mb_size != 0:  # append leftover from split
        nested_idxs += [idxs[chunks * mb_size:]]
    mini_batches = []
    for minibatch_idxs in nested_idxs:
        minibatch = {k: v[minibatch_idxs] for k, v in batch.items()}
        mini_batches.append(minibatch)
    return mini_batches


def to_json(d, indent=2):
    '''Shorthand method for stringify JSON with indent'''
    return json.dumps(d, indent=indent, cls=LabJsonEncoder)


def to_render():
    return os.environ.get('RENDER', 'false') == 'true' or (get_lab_mode() in ('dev', 'enjoy') and os.environ.get('RENDER', 'true') == 'true')


def to_torch_batch(batch, device, is_episodic):
    '''Mutate a batch (dict) to make its values from numpy into PyTorch tensor'''
    for k in batch:
        if is_episodic:  # for episodic format
            batch[k] = np.concatenate(batch[k])
        elif ps.is_list(batch[k]):
            batch[k] = np.array(batch[k])
        batch[k] = torch.from_numpy(batch[k].astype(np.float32)).to(device)
    return batch


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


# Atari image preprocessing


def to_opencv_image(im):
    '''Convert to OpenCV image shape h,w,c'''
    shape = im.shape
    if len(shape) == 3 and shape[0] < shape[-1]:
        return im.transpose(1, 2, 0)
    else:
        return im


def to_pytorch_image(im):
    '''Convert to PyTorch image shape c,h,w'''
    shape = im.shape
    if len(shape) == 3 and shape[-1] < shape[0]:
        return im.transpose(2, 0, 1)
    else:
        return im


def grayscale_image(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


def resize_image(im, w_h):
    return cv2.resize(im, w_h, interpolation=cv2.INTER_AREA)


def normalize_image(im):
    '''Normalizing image by dividing max value 255'''
    # NOTE: beware in its application, may cause loss to be 255 times lower due to smaller input values
    return np.divide(im, 255.0)


def preprocess_image(im, w_h=(84, 84)):
    '''
    Image preprocessing using OpenAI Baselines method: grayscale, resize
    This resize uses stretching instead of cropping
    '''
    im = to_opencv_image(im)
    im = grayscale_image(im)
    im = resize_image(im, w_h)
    im = np.expand_dims(im, 0)
    return im


def debug_image(im):
    '''
    Renders an image for debugging; pauses process until key press
    Handles tensor/numpy and conventions among libraries
    '''
    if torch.is_tensor(im):  # if PyTorch tensor, get numpy
        im = im.cpu().numpy()
    im = to_opencv_image(im)
    im = im.astype(np.uint8)  # typecast guard
    if im.shape[0] == 3:  # RGB image
        # accommodate from RGB (numpy) to BGR (cv2)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imshow('debug image', im)
    cv2.waitKey(0)
