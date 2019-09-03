from Processor import Processor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Evaluation import Evaluation


class ProcessorFPA(Processor):
    def convert_input_output(self, inputs, outputs, sampling_fre):
        if inputs is None:
            return None, None

        data_len = inputs.shape[0]

        return features, outputs

    def white_box_solution(self, sampling_fre):
        # the algorithm
        data_len = self._x_test.shape[0]
        y_pred = np.zeros([data_len])

        gyr_angle_delta = self._x_test[:, 0] / sampling_fre * 100
        err_gyr = self._x_test[:, 1]
        acc_ratio = self._x_test[:, 2]
        err_acc = self._x_test[:, 3]
        for i_sample in range(data_len):
            y_measure_0 = y_pred[i_sample-1] + gyr_angle_delta[i_sample]
            y_measure_1 = acc_ratio[i_sample]
            gain = err_gyr[i_sample] / (err_gyr[i_sample] + err_acc[i_sample])
            y_pred[i_sample] = y_measure_0 + gain * (y_measure_1 - y_measure_0)

        temp = np.zeros([data_len])
        for i_sample in range(data_len):
            temp[i_sample] = temp[i_sample - 1] + gyr_angle_delta[i_sample]

        y_pred = y_pred.reshape([-1, 1])
        # show results
        R2, RMSE, mean_error = Evaluation._get_all_scores(self._y_test, y_pred)
        plt.figure()
        plt.plot(self._y_test)
        # plt.plot(temp)
        # plt.plot(acc_ratio)
        plt.plot(y_pred)
        plt.title('mean error = ' + str(mean_error[0]) + '  RMSE = ' + str(RMSE[0]))
        plt.show()



