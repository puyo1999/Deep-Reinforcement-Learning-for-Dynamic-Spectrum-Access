import os
import sys

#from util.path_util import smart_path
from __init__ import TRAIN_MODES, EVAL_MODES
sys.path.append('../')
import config
from config.spec import spec_util
from model import Agent
from py_lab.experiment.control import Session

# demo.json, ppo_mbr, train/enjoy, ppo_mbr_24_09_17
def get_spec(spec_file, spec_name, lab_mode, pre_):
    if lab_mode in TRAIN_MODES:
        spec = spec_util.get(spec_file, spec_name)
    elif lab_mode == 'enjoy':
        # for enjoy@{session_spec_file}
        # e.g. enjoy@data/reinforce_cartpole_2020_04_13_232521/reinforce_cartpole_t0_s0_spec.json
        session_spec_file = pre_
        assert session_spec_file is not None, 'enjoy mode must specify a `enjoy@{session_spec_file}`'
        spec = util.read(f'{session_spec_file}')
    return spec

# demo.json / ppo_mbr_24_09_17 / true
def get_spec_and_run(spec_file, spec_name, lab_mode):
    print(f'@ get_spec_and_run - spec_file:{spec_file}\nspec-name:{spec_name}\nin mode:{lab_mode}')
    if '@' in lab_mode: #process lab_mode@{predir/prename}
        lab_mode, pre_ = lab_mode.split('@')
    else:
        pre_ = None
    spec = get_spec(spec_file, spec_name, lab_mode, pre_)
    if 'spec_params' not in spec:
        run_spec(spec, lab_mode)

    else: # spec_params 있는 경우
        print()
        param_specs = spec_util.get_param_specs(spec)
        #search.run_param_specs(param_specs)

def run_spec(spec, lab_mode):
    os.environ['lab_mode'] = lab_mode
    spec = spec_util.override_spec(spec, lab_mode)
    print(f'lab_mode:{lab_mode}')
    if lab_mode in TRAIN_MODES:
        spec_util.save(spec)

        Agent(spec)

        Session(spec).run()
    else:
        raise ValueError(f'Unrecognizable lab_mode not of {TRAIN_MODES}')



def main():

    args = sys.argv[1:]

    print(f'args len : {len(args)}')
    if len(args) <= 1:
        print(f'only 1 args')
        job_file = args[0] if len(args)==1 else "job/experiments.json"
    else:
        print()
        assert len(args) == 3, f'To use sys args, specify spec_file, spec_name, lab_mode'

        #
        # demo.json ppo_mbr_24_09_17 train
        get_spec_and_run(*args)


if __name__ == '__main__':
    main()