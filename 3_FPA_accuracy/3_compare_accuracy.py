import pandas as pd
import numpy as np
from PaperFigures import ErrorBarFigure
import matplotlib.pyplot as plt
from const import SUB_NAMES, TRIAL_NAMES


for sub_name in SUB_NAMES:
    for trial_name in TRIAL_NAMES:
        file_path = 'step_result/' + sub_name + '/step_result_of' + trial_name + '.csv'
        trial_result = pd.read_csv(file_path, index_col=False)


# def get_step_result()
#
# detailed_result_df = pd.read_csv('detailed_result_df.csv', index_col=False)
# ErrorBarFigure.draw_true_esti_compare_figure(detailed_result_df)
# ErrorBarFigure.draw_error_bar_figure_trials(detailed_result_df)
# ErrorBarFigure.draw_error_bar_figure_subtrials(detailed_result_df)
# plt.show()






