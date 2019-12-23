import pandas as pd
import numpy as np
from PaperFigures import ErrorBarFigure, PaperFigure
import matplotlib.pyplot as plt
from const import SUB_NAMES, TRIAL_NAMES, FPA_TRIALS, FPA_NAME_LIST
from ResultReader import ResultReader


compensation_tbme, compensation_acc_ratio = 15, 6.5  # compensate the bias
result_df_column_names = FPA_NAME_LIST + ['subtrial_id', 'subject_id', 'trial_id']
result_all_df = pd.DataFrame(columns=result_df_column_names)
for sub_name in SUB_NAMES[:12]:
    print('\nSubject: ' + sub_name)
    sub_id = SUB_NAMES.index(sub_name)
    for trial_name in FPA_TRIALS:
        print('Trial: ' + trial_name)
        trial_id = TRIAL_NAMES.index(trial_name)
        file_path = '../2_FPA/step_result/' + sub_name + '/step_result_of_' + trial_name + '.csv'
        trial_result_sample = pd.read_csv(file_path, index_col=False)
        steps = ResultReader.get_steps(trial_result_sample)
        trial_result = ResultReader.get_fpas(FPA_NAME_LIST, trial_result_sample, steps)
        sub_id_array = np.full([trial_result.shape[0]], sub_id)
        trial_id_array = np.full([trial_result.shape[0]], trial_id)
        trial_result = np.column_stack([trial_result, sub_id_array, trial_id_array])

        trial_result_df = pd.DataFrame(trial_result, columns=result_df_column_names)
        result_all_df = result_all_df.append(trial_result_df, ignore_index=True)

result_all_df[FPA_NAME_LIST[1]] = result_all_df[FPA_NAME_LIST[1]] - compensation_tbme
result_all_df[FPA_NAME_LIST[2]] = result_all_df[FPA_NAME_LIST[2]] - compensation_acc_ratio


# PaperFigure.each_sub_fig(result_all_df)

ErrorBarFigure.show_each_pair(result_all_df)
ErrorBarFigure.compare_mean_error(result_all_df, 'subject_id', x_label='subject id')
ErrorBarFigure.compare_mean_error(result_all_df, 'trial_id', ['1.0 m/s', '1.2 m/s', '1.4 m/s'], 'walking speed')
subtrial_name_list = ['B/L - 10', 'B/L - 5', 'B/L', 'B/L + 15', 'B/L + 30']
ErrorBarFigure.compare_mean_error(result_all_df, 'subtrial_id', subtrial_name_list, 'feedback value')
ErrorBarFigure.compare_mean_value(result_all_df, 'subtrial_id', subtrial_name_list, 'feedback value')
plt.show()


#
# detailed_result_df = pd.read_csv('detailed_result_df.csv', index_col=False)
# ErrorBarFigure.draw_true_esti_compare_figure(detailed_result_df)
# ErrorBarFigure.draw_error_bar_figure_trials(detailed_result_df)
# ErrorBarFigure.draw_error_bar_figure_subtrials(detailed_result_df)
# plt.show()






