"""
# Note that this is a basic example for you to understand how to implement the algorithm
"""

from Processor import Processor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Evaluation import Evaluation
from const import PROCESSED_DATA_PATH
from scipy.signal import lfilter, butter


class ProcessorTrunk(Processor):
    def __init__(self, train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, grf_side, param_name,
                 IMU_location, data_type, do_input_norm=True, do_output_norm=False, show_plots=True):
        super().__init__(train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, grf_side, param_name,
                         IMU_location, data_type, do_input_norm=do_input_norm, do_output_norm=do_output_norm,
                         show_plots=show_plots)
        self.comp_filter_mix_ratio_ml = .007
        self.comp_filter_mix_ratio_ap = .01
        self.filter_order = 4
        self.filter_cutoff_freq = 6

    def convert_input_output(self, inputs, outputs, id_df, sampling_fre):
        if inputs is None:
            return None, None
        
        trial_length = inputs.shape[0]
        estimated_angle = np.zeros([trial_length, 1])

        filter_order = self.filter_order
        cut_off_fre = self.filter_cutoff_freq
        fre = cut_off_fre / (sampling_fre / 2)
        b, a = butter(filter_order, fre, 'lowpass')
        filtered_acc = lfilter(b, a, inputs[:, 0:3], axis=0)

        # see the id_df
        if self.param_name == "trunk_ap_angle":
            comp_filter_param = self.comp_filter_mix_ratio_ap
            calibration_angle = self.cal_angle
            print(calibration_angle, end=',')
            bias = calibration_angle * .36 + 5
            acc_angle_prev = (-((np.arctan2(np.linalg.norm(inputs[0, :]), inputs[0, 2]) / np.pi * 180) - 90) - 
                              calibration_angle) + bias
            estimated_angle[0] = acc_angle_prev
            for i in range(1, trial_length):
                acc_angle = (-((np.arctan2(np.linalg.norm(filtered_acc[i, :]), filtered_acc[i, 2]) / np.pi * 180) - 90)
                             - calibration_angle) + bias
                gyro_angle = estimated_angle[i-1] - (inputs[i, 4] / sampling_fre / np.pi * 180)
                estimated_angle[i] = acc_angle * comp_filter_param + (1-comp_filter_param) * gyro_angle

        elif self.param_name == "trunk_ml_angle":
            comp_filter_param = self.comp_filter_mix_ratio_ml
            calibration_angle = self.cal_angle
            bias = .73
            print(calibration_angle, end=',')
            acc_angle_prev = ((np.arctan2(np.linalg.norm(inputs[0, :]), inputs[0, 1]) / np.pi * 180) - 90) + \
                calibration_angle - bias
            estimated_angle[0] = acc_angle_prev
            for i in range(1, trial_length):
                acc_angle = (np.arctan2(np.linalg.norm(filtered_acc[i, :]), filtered_acc[i, 1]) / np.pi * 180) - 90 + \
                    calibration_angle - bias
                gyro_angle = estimated_angle[i-1] - (inputs[i, 5] / sampling_fre / np.pi * 180)
                estimated_angle[i] = acc_angle * comp_filter_param + (1 - comp_filter_param) * gyro_angle

        estimated_angle = estimated_angle.reshape([-1, 1])
        if self.show_plots:
            plt.figure()
            plt.plot(range(trial_length), inputs[:, 0], marker='o', markerfacecolor="blue")
            plt.plot(range(trial_length), filtered_acc[:, 0], marker='.', markerfacecolor="orange")
            plt.show()

            plt.figure()
            plt.plot(range(trial_length), estimated_angle, marker='o', markerfacecolor="blue")
            plt.plot(range(trial_length), outputs, marker='.', markerfacecolor="orange")
            plt.show()
        return estimated_angle, outputs

    def calibrate_subject(self, subject_name):

        # get static trial data
        static_file = PROCESSED_DATA_PATH + '\\' + subject_name + '\\' + '200Hz\\static.csv'
        data_static = pd.read_csv(static_file)
        column_names_acc = ["trunk_acc_x", "trunk_acc_y", "trunk_acc_z"]
        acc_data = data_static[column_names_acc]
        acc_vals = np.mean(acc_data)
        acc_angle = 0
        if self.param_name == "trunk_ap_angle":
            acc_angle = -(np.arctan2(-acc_vals[0], acc_vals[2]) / np.pi * 180) + 90
        elif self.param_name == "trunk_ml_angle":
            acc_angle = -(np.arctan2(-acc_vals[0], acc_vals[1]) / np.pi * 180) + 90
        self.cal_angle = acc_angle

    def white_box_solution(self):
        y_pred = self._x_test
        R2, RMSE, mean_error = Evaluation._get_all_scores(self._y_test, y_pred)

        # show results
        if self.show_plots:
            plt.figure()
            plt.plot(self._y_test, y_pred, '.')
            plt.title('mean error = ' + str(mean_error[0]) + '  RMSE = ' + str(RMSE[0]))
            plt.ylabel('predicted angle')
            plt.xlabel('true trunk anterior-posterior angle')
            plt.show()
        print(mean_error[0], end=',')
        print(R2[0])

