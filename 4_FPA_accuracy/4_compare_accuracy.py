import pandas as pd
import numpy as np
from PaperFigures import ErrorBarFigure
from ProcessorFPAHS import InitFPA
import matplotlib.pyplot as plt
from const import SUB_NAMES_HS, TRIAL_NAMES_HS, FPA_NAME_LIST_HS
from ResultReader import ResultReader


test_date = '0227'
# ErrorBarFigure.compare_acc(sub_id=6)
# plt.show()

num_of_steps, steps_to_skip = 51, 20
result_raw_df = pd.read_csv('../3_FPA_Haisheng/result_conclusion/' + test_date + '/step_result_hanning.csv', index_col=False)
result_df_column_names = FPA_NAME_LIST_HS + ['sub_name', 'trial_id']
result_all_df = pd.DataFrame(columns=result_df_column_names)
for sub_name in SUB_NAMES_HS[:]:
    print('\nSubject: ' + sub_name)
    sub_id = SUB_NAMES_HS.index(sub_name)
    for trial_id in range(len(TRIAL_NAMES_HS)):
        trial_raw_df = result_raw_df[(result_raw_df['sub_name'] == sub_name) & (result_raw_df['trial_id'] == trial_id)]
        steps = ResultReader.get_steps(trial_raw_df)
        trial_result = ResultReader.get_fpas(FPA_NAME_LIST_HS, trial_raw_df, steps, num_of_steps, steps_to_skip)
        sub_id_array = np.full([trial_result.shape[0]], sub_id)
        trial_id_array = np.full([trial_result.shape[0]], trial_id)
        trial_result = np.column_stack([trial_result, sub_id_array, trial_id_array])

        trial_result_df = pd.DataFrame(trial_result, columns=result_df_column_names)
        result_all_df = result_all_df.append(trial_result_df, ignore_index=True)

# """Figures for checking results"""
# ErrorBarFigure.compare_mean_error_of_mean_sub_error(result_all_df, 'trial_id', TRIAL_NAMES_HS, 'trial name')
# ErrorBarFigure.show_each_pair(result_all_df)
# ErrorBarFigure.compare_mean_error(result_all_df, 'sub_name', x_label='subject name')
# ErrorBarFigure.compare_mean_error(result_all_df, 'trial_id', TRIAL_NAMES_HS, 'trial name')
# ErrorBarFigure.compare_mean_value(result_all_df, 'trial_id', TRIAL_NAMES_HS, 'trial name')

"""Figures for new paper"""
# ErrorBarFigure.compare_mean_value_ASB(result_all_df)
ErrorBarFigure.compare_mean_value_paper(result_all_df)
ErrorBarFigure.compare_mean_value_paper_no_pvalue(result_all_df)
# ErrorBarFigure.compare_accuracy_paper(result_all_df)
ErrorBarFigure.print_accuracy_paper(result_all_df)
ErrorBarFigure.print_accuracy_overall(result_all_df, digits=1)
plt.show()

























