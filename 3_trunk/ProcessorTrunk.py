"""
# Note that this is a basic example for you to understand how to implement the algorithm
"""

from Processor import Processor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Evaluation import Evaluation


class ProcessorTrunk(Processor):
    def convert_input_output(self, inputs, outputs, id_df, sampling_fre):
        if inputs is None:
            return None, None

        # see the id_df

        # convert data into features
        feature_0 = - np.arctan2(inputs[:, 2], np.linalg.norm(inputs[:, 0:3], axis=1)) / np.pi * 180
        feature_0 = feature_0.reshape([-1, 1])
        return feature_0, outputs

    def white_box_solution(self):
        # the algorithm
        y_pred = self._x_test

        # show results
        R2, RMSE, mean_error = Evaluation._get_all_scores(self._y_test, y_pred)
        plt.figure()
        plt.plot(self._y_test, y_pred, '.')
        plt.title('mean error = ' + str(mean_error[0]) + '  RMSE = ' + str(RMSE[0]))
        plt.ylabel('predicted angle')
        plt.xlabel('true trunk anterior-posterior angle')
        plt.show()


