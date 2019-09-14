"""
Export FIR filter parameter for C++ code
"""
import numpy as np
from scipy import signal
from const import TRIAL_START_BUFFER, FILTER_WIN_LEN
import json


def write_text_file(path, text):
    """Write a string to a file"""
    with open(path, "w") as text_file:
        print(text, file=text_file)


# def save_filter_param(cut_off_fre, param_name, strike_delay, off_delay):
off_delay = 10
strike_delay = 10
param_name = 'FPA'
sampling_fre = 200
filter_win_len = 100
param_file = 'filter_param_files/filter_param_' + param_name + '.json'

a = 1

wn_gyr_integration = 6 / (sampling_fre/2)
b_gyr_integration = signal.firwin(filter_win_len, wn_gyr_integration)

wn_strike_off = 5 / (sampling_fre/2)
b_strike_off = signal.firwin(filter_win_len, wn_strike_off)

wn_acc_ratio = 2 / (sampling_fre/2)
b_acc_ratio = signal.firwin(filter_win_len, wn_acc_ratio)

filter_delay = int(FILTER_WIN_LEN / 2)

filter_param = {'b_gyr_integration': b_gyr_integration.tolist(), 'b_strike_off': b_strike_off.tolist(),
                'b_acc_ratio': b_acc_ratio.tolist(), 'a': a, 'filter_win_len': filter_win_len,
                'filter_delay': filter_delay, 'strike_delay': strike_delay, 'off_delay': off_delay,
                'start_buffer': TRIAL_START_BUFFER}
with open(param_file, 'w') as param_file:
    print(json.dumps(filter_param, sort_keys=True, indent=4, separators=(',', ': ')), file=param_file)



























