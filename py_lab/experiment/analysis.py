from py_lab.lib import logger, util
from py_lab.spec import random_baseline

import numpy as np
import pandas as pd
import pydash as ps
import shutil
import torch
import warnings

METRICS_COLS = [
    'final_return_ma',
    'strength', 'max_strength', 'final_strength',
    'sample_efficiency', 'training_efficiency',
    'stability', 'consistency',
]

logger = logger.get_logger(__name__)


# methods to generate returns (total rewards)
def calc_session_metrics(session_df, env_name, info_prepath=None, df_mode=None):
    '''
    Calculate the session metrics: strength, efficiency, stability
    @param DataFrame:session_df Dataframe containing reward, frame, opt_step
    @param str:env_name Name of the environment to get its random baseline
    @param str:info_prepath Optional info_prepath to auto-save the output to
    @param str:df_mode Optional df_mode to save with info_prepath
    @returns dict:metrics Consists of scalar metrics and series local metrics
    '''
    rand_bl = random_baseline.get_random_baseline(env_name)
    if rand_bl is None:
        mean_rand_returns = 0.0
        logger.warn('Random baseline unavailable for environment. Please generate separately.')
    else:
        mean_rand_returns = rand_bl['mean']
    mean_returns = session_df['total_reward']
    frames = session_df['frame']
    opt_steps = session_df['opt_step']

    final_return_ma = mean_returns[-viz.PLOT_MA_WINDOW:].mean()
    str_, local_strs = calc_strength(mean_returns, mean_rand_returns)
    max_str, final_str = local_strs.max(), local_strs.iloc[-1]
    with warnings.catch_warnings():  # mute np.nanmean warning
        warnings.filterwarnings('ignore')
        sample_eff, local_sample_effs = calc_efficiency(local_strs, frames)
        train_eff, local_train_effs = calc_efficiency(local_strs, opt_steps)
        sta, local_stas = calc_stability(local_strs)

    # all the scalar session metrics
    scalar = {
        'final_return_ma': final_return_ma,
        'strength': str_,
        'max_strength': max_str,
        'final_strength': final_str,
        'sample_efficiency': sample_eff,
        'training_efficiency': train_eff,
        'stability': sta,
    }
    # all the session local metrics
    local = {
        'mean_returns': mean_returns,
        'strengths': local_strs,
        'sample_efficiencies': local_sample_effs,
        'training_efficiencies': local_train_effs,
        'stabilities': local_stas,
        'frames': frames,
        'opt_steps': opt_steps,
    }
    metrics = {
        'scalar': scalar,
        'local': local,
    }
    if info_prepath is not None:  # auto-save if info_prepath is given
        util.write(metrics, f'{info_prepath}_session_metrics_{df_mode}.pkl')
        util.write(scalar, f'{info_prepath}_session_metrics_scalar_{df_mode}.json')
        # save important metrics in info_prepath directly
        util.write(scalar, f'{info_prepath.replace("info/", "")}_session_metrics_scalar_{df_mode}.json')
    return metrics

def analyze_session(session_spec, session_df, df_mode, plot=True):
    '''Analyze session and save data, then return metrics. Note there are 2 types of session_df: body.eval_df and body.train_df'''
    info_prepath = session_spec['meta']['info_prepath']
    session_df = session_df.copy()  # prevent modification
    assert len(session_df) > 2, f'Need more than 2 datapoint to calculate metrics'  # first datapoint at frame 0 is empty
    util.write(session_df, util.get_session_df_path(session_spec, df_mode))
    # calculate metrics
    session_metrics = calc_session_metrics(session_df, ps.get(session_spec, 'env.0.name'), info_prepath, df_mode)
    if plot:
        # plot graph
        viz.plot_session(session_spec, session_metrics, session_df, df_mode)
        viz.plot_session(session_spec, session_metrics, session_df, df_mode, ma=True)
    return session_metrics