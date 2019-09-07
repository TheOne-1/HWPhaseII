from const import SUB_NAMES, PROCESSED_DATA_PATH
import numpy as np
import pandas as pd
from ProcessorTest import get_angles_via_gra_mag


def initialize_static_offset(subject_id, side):
    subject_name = SUB_NAMES[subject_id]
    # get static trial data
    static_marker_file = PROCESSED_DATA_PATH + '\\' + subject_name + '\\' + '200Hz\\static.csv'
    data_static = pd.read_csv(static_marker_file)

    # get marker offset
    marker_column_names = [side.upper() + marker_name + '_' + axis_name for marker_name in ['FM2', 'FCC']
                           for axis_name in ['x', 'y']]
    toe_heel_static = data_static[marker_column_names]
    data_static_marker = np.mean(toe_heel_static)
    delta_x = data_static_marker[0] - data_static_marker[2]
    delta_y = data_static_marker[1] - data_static_marker[3]
    yaw_marker = np.rad2deg(np.arctan2(delta_x, delta_y))

    # get xsens via sensor output
    yaw_xsens_raw = data_static[side + '_foot_yaw']
    yaw_xsens = - np.mean(yaw_xsens_raw) - 180
    yaw_diff_degree = yaw_xsens - yaw_marker

    # get xsens via gravity and mag
    print(SUB_NAMES[i_sub], end='\t\t\t\t')

    euler_angles = get_angles_via_gra_mag(data_static, 'l_foot')
    yaw_diff_degree = np.mean(euler_angles[:, 2]) + 180

    return yaw_xsens, yaw_marker, yaw_diff_degree


angle_bias = []
for i_sub in range(19):
    yaw_xsens, yaw_marker, yaw_diff_degree = initialize_static_offset(i_sub, 'l')

    if abs(yaw_diff_degree) < 30:
        angle_bias.append(yaw_diff_degree)
    else:
        angle_bias.append(0)

    # print(SUB_NAMES[i_sub], end='\t\t\t\t')
    # print([yaw_xsens, yaw_marker, yaw_diff_degree])



