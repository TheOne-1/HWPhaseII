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


def save_filter_param(cut_off_fre, param_name, strike_delay, off_delay):
    sampling_fre = 200
    filter_win_len = 100
    param_file = 'filter_param_files/filter_param_' + param_name + '.json'
    wn = cut_off_fre / (sampling_fre/2)
    b = signal.firwin(filter_win_len, wn)
    a = 1

    filter_delay = int(FILTER_WIN_LEN / 2)

    filter_param = {'wn': wn, 'b': b.tolist(), 'a': a, 'filter_win_len': filter_win_len,
                    'filter_delay': filter_delay, 'strike_delay': strike_delay, 'off_delay': off_delay,
                    'start_buffer': TRIAL_START_BUFFER}
    with open(param_file, 'w') as param_file:
        print(json.dumps(filter_param, sort_keys=True, indent=4, separators=(',', ': ')), file=param_file)


save_filter_param(5, 'lr_si', 8, 6)
save_filter_param(5, 'FPA_strike_off', 10, 10)
save_filter_param(6, 'FPA_gyr_integration', 0, 0)
save_filter_param(2, 'FPA_acc_peak', 0, 0)

























