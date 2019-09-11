"""
# Note that this is a basic example for you to understand how to implement the algorithm
"""

from Processor import Processor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Evaluation import Evaluation
from const import PROCESSED_DATA_PATH, SUB_NAMES


class ProcessorTrunk(Processor):
    def __init__(self, train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, grf_side, param_name,
                 IMU_location, data_type, do_input_norm=True, do_output_norm=False, show_plots=True):
        super().__init__(train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, grf_side, param_name,
                 IMU_location, data_type, do_input_norm=do_input_norm, do_output_norm=do_output_norm)
        self.paramA = .1






    def convert_input_output(self, inputs, outputs, id_df, sampling_fre):
        if inputs is None:
            return None, None
        trial_length = inputs.shape[0]
        acc_filtered = np.zeros([trial_length, 3])
        feature_0 = np.zeros([trial_length, 1])
        bias = 0
        # see the id_df
        if self.param_name == "trunk_ap_angle":
            comp_filter_param = self.paramA
            exp_smooth_param = 0.02
            scaling_factor = 1
            # Initialization values for Comp Filter
            calibration_angle = self.cal_angle
            print(calibration_angle, end=',')
            bias = calibration_angle * .4408 + 7.1926
            #calibration_angle = 0
            #bias = 0
            print(bias, end=',')
            acc_angle_prev = (-((np.arctan2(-inputs[0, 0], inputs[0, 2]) / np.pi * 180) - 90) - calibration_angle)/scaling_factor + bias
            acc_prev = inputs[0, 0:3]

            filter_result_angle_prev = acc_angle_prev

            feature_0[0] = filter_result_angle_prev
            acc_filtered[0, :] = acc_prev

            for i in range(1, trial_length):
                filtered_acc = acc_prev * (1-exp_smooth_param) + inputs[i, 0:3] * exp_smooth_param

                acc_angle = (-((np.arctan2(-filtered_acc[0], filtered_acc[2]) / np.pi * 180) - 90) - calibration_angle)/scaling_factor + bias

                gyro_angle = filter_result_angle_prev - (inputs[i, 4] / sampling_fre / np.pi * 180)

                filter_result_angle = acc_angle * comp_filter_param + (1-comp_filter_param) * gyro_angle

                acc_prev = filtered_acc
                gyro_prev = inputs[i, 3:6]
                acc_angle_prev = acc_angle
                gyro_angle_prev = gyro_angle
                filter_result_angle_prev = filter_result_angle
                feature_0[i] = filter_result_angle
                acc_filtered[i, :] = filtered_acc
        elif self.param_name == "trunk_ml_angle":
            comp_filter_param = .007
            exp_smooth_param = 0.015
            # Initialization values for Comp Filter
            calibration_angle = self.cal_angle

            print(calibration_angle, end=',')
            gyro_angle_prev = 0
            acc_angle_prev = (np.arctan2(-inputs[0, 0], inputs[0, 1]) / np.pi * 180)/scaling_factor - 90 + calibration_angle
            acc_prev = inputs[0, 0:3]
            gyro_prev = inputs[0, 3:6]

            filter_result_angle_prev = acc_angle_prev

            feature_0[0] = filter_result_angle_prev - bias
            acc_filtered[0, :] = acc_prev

            for i in range(1, trial_length):
                filtered_acc = acc_prev * (1 - exp_smooth_param) + inputs[i, 0:3] * exp_smooth_param

                acc_angle = (np.arctan2(-filtered_acc[0], filtered_acc[1]) / np.pi * 180)/scaling_factor - 90 + calibration_angle

                gyro_angle = filter_result_angle_prev - (inputs[i, 5] / sampling_fre / np.pi * 180)

                filter_result_angle = acc_angle * comp_filter_param + (1 - comp_filter_param) * gyro_angle

                acc_prev = filtered_acc
                gyro_prev = inputs[i, 3:6]
                acc_angle_prev = acc_angle
                gyro_angle_prev = gyro_angle
                filter_result_angle_prev = filter_result_angle
                feature_0[i] = filter_result_angle
                acc_filtered[i, :] = filtered_acc

        feature_0 = feature_0.reshape([-1, 1])
        if self.show_plots:
            plt.figure()
            plt.plot(range(trial_length), inputs[:, 0], marker='o', markerfacecolor="blue")
            plt.plot(range(trial_length), acc_filtered[:, 0], marker='.', markerfacecolor="orange")
            plt.show()
            #
            # plt.figure()
            # plt.plot(range(trial_length), inputs[:, 4], marker='o', markerfacecolor="blue")
            # plt.show()

            plt.figure()
            plt.plot(range(trial_length), feature_0, marker='o', markerfacecolor="blue")
            plt.plot(range(trial_length), outputs, marker='.', markerfacecolor="orange")
            plt.show()
        return feature_0, outputs

    def calibrate_subject(self, subject_name):

        # get static trial data
        static_file = PROCESSED_DATA_PATH + '\\' + subject_name + '\\' + '200Hz\\static.csv'
        data_static = pd.read_csv(static_file)

        # get marker offset
        column_names_acc = ["trunk_acc_x", "trunk_acc_y", "trunk_acc_z"]

        acc_data = data_static[column_names_acc]
        acc_vals = np.mean(acc_data)
        if self.param_name == "trunk_ap_angle":
            acc_angle = -(np.arctan2(-acc_vals[0], acc_vals[2]) / np.pi * 180) + 90
        elif self.param_name == "trunk_ml_angle":
            acc_angle = -(np.arctan2(-acc_vals[0], acc_vals[1]) / np.pi * 180) + 90
        self.cal_angle = acc_angle

    def white_box_solution(self):
        # the algorithm
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

