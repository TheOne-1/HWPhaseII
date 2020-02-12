import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from const import LINE_WIDTH, FONT_DICT_SMALL, FONT_SIZE, FPA_NAME_LIST_HS, SUB_NAMES, FONT_DICT, SUB_NAMES_HS, \
    TRIAL_NAMES_HS, FONT_DICT_X_SMALL, DATA_PATH_HS
from Evaluation import Evaluation
from scipy.stats import ttest_ind
from ProcessorFPAHS import InitFPA
from DataProcessorHS import DataInitializerHS
from numpy import sin, cos
from Processor import Processor


class BaseFigure:
    pass

    @staticmethod
    def format_plot():
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_tick_params(width=LINE_WIDTH)
        ax.yaxis.set_tick_params(width=LINE_WIDTH)
        ax.spines['left'].set_linewidth(LINE_WIDTH)
        ax.spines['bottom'].set_linewidth(LINE_WIDTH)

    @staticmethod
    def each_sub_summary(result_all_df):
        sub_names = list(set(result_all_df['sub_name'].values))
        predict_result_df = pd.DataFrame()
        for sub_name in sub_names:
            sub_id = SUB_NAMES_HS.index(sub_name)
            sub_result_df = result_all_df[result_all_df['subject_id'] == sub_name]
            pearson_coeff, RMSE, mean_error = Evaluation.plot_fpa_result(
                sub_result_df['FPA_true'], sub_result_df['FPA_estis'], sub_id)
            predict_result_df = Evaluation.insert_prediction_result(
                predict_result_df, SUB_NAMES[int(sub_name)], pearson_coeff, RMSE, mean_error)
        Evaluation.export_prediction_result(predict_result_df)


class ErrorBarFigure(BaseFigure):
    @staticmethod
    def compare_acc(sub_id):
        """
        Show the average step. All the steps were interpolated to 100 samples.
        :return:
        """
        def init_trial(trial_id):
            gait_data_path = base_path + TRIAL_NAMES_HS[trial_id] + '.csv'
            gait_data_df = pd.read_csv(gait_data_path, index_col=False)

            gait_param_path = base_path + 'param_of_' + TRIAL_NAMES_HS[trial_id] + '.csv'
            gait_param_df = pd.read_csv(gait_param_path, index_col=False)
            steps, stance_phase_flag = InitFPA.initalize_steps_and_stance_phase(gait_param_df)
            euler_angles_esti = InitFPA.get_euler_angles_gradient_decent_from_stance(
                gait_data_df, stance_phase_flag)
            acc_IMU_rotated = InitFPA.get_rotated_acc(placement_R_foot_sensor, gait_data_df, euler_angles_esti)

            acc_x_step_array, acc_y_step_array = np.zeros([len(steps) - 6, 100]), np.zeros([len(steps) - 6, 100])
            for i_step in range(3, len(steps) - 3):
                last_off = steps[i_step][1]
                current_strike = steps[i_step + 1][0]
                acc_x_step_array[i_step - 3, :] = Processor.resample_channel(acc_IMU_rotated[last_off:current_strike, 0], 100)
                acc_y_step_array[i_step - 3, :] = Processor.resample_channel(acc_IMU_rotated[last_off:current_strike, 1], 100)

            return np.mean(acc_x_step_array, axis=0), np.mean(acc_y_step_array, axis=0)

        sub_folder = SUB_NAMES_HS[sub_id]
        base_path = DATA_PATH_HS + 'processed/' + sub_folder + '/'
        placement_offset = DataInitializerHS.init_placement_offset(sub_folder)
        offset_rad = - np.deg2rad(placement_offset)
        placement_R_foot_sensor = np.array([
            [cos(offset_rad), sin(offset_rad), 0],
            [-sin(offset_rad), cos(offset_rad), 0],
            [0, 0, 1]])

        acc_x_toe_in, acc_y_toe_in = init_trial(3)
        acc_x_toe_out, acc_y_toe_out = init_trial(6)

        plt.figure(figsize=(8, 10))
        plt.subplot(211)
        acc_x_plot, = plt.plot(acc_x_toe_in)
        acc_y_plot, = plt.plot(acc_y_toe_in)
        plt.subplot(212)
        acc_x_plot, = plt.plot(acc_x_toe_out)
        acc_y_plot, = plt.plot(acc_y_toe_out)
        legend_names = ['x-axis', 'y-axis']
        plt.legend([acc_x_plot, acc_y_plot], legend_names, fontsize=FONT_SIZE, frameon=True,
                   bbox_to_anchor=(0.5, 0.75))


    @staticmethod
    def compare_mean_value_paper(result_all_df):
        """Only compare acc ratio and true FPAs"""
        x_locs = [3, 2, 1, 0, 4, 5, 6]
        values_acc_ratio = result_all_df['FPA_estis']
        values_vicon = result_all_df['FPA_true']
        pvalue_list, means_vicon, stds_vicon, means_acc_ratio, stds_acc_ratio, id_list = \
            ErrorBarFigure.mean_value_ttest(values_vicon, values_acc_ratio, result_all_df, np.mean)

        plt.figure(figsize=(14, 6))
        ErrorBarFigure.format_plot()
        bars = []
        for i_cate in range(len(id_list)):
            bars.append(plt.bar(x_locs[i_cate], means_vicon[i_cate], color='grey', width=0.35))
            ErrorBarFigure.draw_half_ebars(means_vicon[i_cate], stds_vicon[i_cate], x_locs[i_cate],
                                           bool(np.sign(means_vicon[i_cate])+1))

        for i_cate in range(len(id_list)):
            bars.append(plt.bar(x_locs[i_cate] + 0.35, means_acc_ratio[i_cate], color='brown', width=0.35))
            ErrorBarFigure.draw_half_ebars(means_acc_ratio[i_cate], stds_acc_ratio[i_cate], x_locs[i_cate] + 0.35,
                                           bool(np.sign(means_acc_ratio[i_cate]) + 1))
        ax = plt.gca()
        x_tick_loc = [i + 0.175 for i in range(len(id_list))]
        ax.tick_params(labelsize=FONT_DICT['fontsize'])
        ax.set_ylabel('Foot Progression Angle (deg)', fontdict=FONT_DICT)
        ax.set_xticks(x_tick_loc)
        x_tick_list = ['Large Toe-in', 'Medium Toe-in', 'Small Toe-in', 'Normal', 'Small Toe-out', 'Medium Toe-out', 'Large Toe-out']
        ax.set_xticklabels(x_tick_list, fontdict=FONT_DICT_X_SMALL)
        ax.set_yticks(range(-30, 41, 10))
        y_tick_list = ['-30', '-20', '-10', '0', '10', '20', '30', '40']
        ax.set_yticklabels(y_tick_list, fontdict=FONT_DICT)
        ax.set_ylim(-33, 40)
        ax.set_xlim(-0.5, 6.86)
        plt.plot([-0.5, 6.86], [0, 0], color='black')
        plt.legend([bars[0], bars[-1]], ['FPA: Motion Capture', 'FPA: IMU (Proposed New Method)'], handlelength=2,
                   handleheight=1.3, bbox_to_anchor=(0.43, 1.03), ncol=1, fontsize=FONT_SIZE, frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 1])

    # @staticmethod
    # def print_accuracy_paper(result_all_df):

    @staticmethod
    def compare_accuracy_paper(result_all_df):
        x_locs = [3, 2, 1, 0, 4, 5, 6]
        error_tbme = result_all_df['FPA_true'] - result_all_df['FPA_tbme']
        error_acc_ratio = result_all_df['FPA_true'] - result_all_df['FPA_estis']
        pvalue_list, RMSE_tbme, stds_tbme, RMSE_acc_ratio, stds_acc_ratio, id_list = \
            ErrorBarFigure.mean_value_ttest(error_tbme, error_acc_ratio, result_all_df, np.mean)       # ErrorBarFigure.root_mean_fun

        plt.figure(figsize=(14, 6))
        ErrorBarFigure.format_plot()
        bars = []
        for i_cate in range(len(id_list)):
            bars.append(plt.bar(x_locs[i_cate], RMSE_tbme[i_cate], color='orange', width=0.35))
            ErrorBarFigure.draw_half_ebars(RMSE_tbme[i_cate], stds_tbme[i_cate], x_locs[i_cate],
                                           bool(np.sign(RMSE_tbme[i_cate])+1))

        for i_cate in range(len(id_list)):
            bars.append(plt.bar(x_locs[i_cate] + 0.35, RMSE_acc_ratio[i_cate], color='brown', width=0.35))
            ErrorBarFigure.draw_half_ebars(RMSE_acc_ratio[i_cate], stds_acc_ratio[i_cate], x_locs[i_cate] + 0.35,
                                           bool(np.sign(RMSE_acc_ratio[i_cate]) + 1))
        ax = plt.gca()
        x_tick_loc = [i + 0.175 for i in range(len(id_list))]
        ax.tick_params(labelsize=FONT_DICT['fontsize'])
        ax.set_ylabel('Mean Error of FPA Estimation (deg)', fontdict=FONT_DICT)
        ax.set_xticks(x_tick_loc)
        x_tick_list = ['Large Toe-in', 'Medium Toe-in', 'Small Toe-in', 'Normal', 'Small Toe-out', 'Medium Toe-out', 'Large Toe-out']
        ax.set_xticklabels(x_tick_list, fontdict=FONT_DICT_X_SMALL)
        ax.set_ylim(-6.2, 6.2)
        ax.set_yticks(range(-4, 7, 2))
        y_tick_list = ['-4', '-2', '0', '2', '4', '6']
        ax.set_yticklabels(y_tick_list, fontdict=FONT_DICT)
        ax.set_xlim(-0.5, 6.86)
        plt.plot([-0.5, 6.86], [0, 0], color='black')
        plt.legend([bars[0], bars[-1]], ['Algorithm Proposed in [xx]', 'Algorithm Proposed in This Paper)'], handlelength=2,
                   handleheight=1.3, bbox_to_anchor=(0.52, 1.03), ncol=1, fontsize=FONT_SIZE, frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 1])

    @staticmethod
    def root_mean_fun(data):
        output_data = np.sqrt(np.average(data ** 2, axis=0))
        return output_data

    # @staticmethod
    # def pearson_value_ttest(data):

    @staticmethod
    def mean_value_ttest(true_values, predicted_values, result_all_df, raw_data_operate_fun):
        pvalue_list = []
        true_means, true_stds = [], []
        pred_means, pred_stds = [], []
        trial_ids = result_all_df['trial_id']
        sub_names = result_all_df['sub_name']
        trial_id_list = list(set(trial_ids))
        sub_name_list = list(set(sub_names))
        trial_id_list.sort()
        for trial_id in trial_id_list:
            trial_true = true_values[trial_ids == trial_id]
            trial_pred = predicted_values[trial_ids == trial_id]
            true_sub_results, pred_sub_results = [], []
            for sub_name in sub_name_list:
                true_sub_values = trial_true[sub_names == sub_name]
                pred_sub_values = trial_pred[sub_names == sub_name]
                true_sub_results.append(raw_data_operate_fun(true_sub_values))
                pred_sub_results.append(raw_data_operate_fun(pred_sub_values))

            _, pvalue = ttest_ind(true_sub_results, pred_sub_results)
            print(TRIAL_NAMES_HS[int(trial_id)] + ', p value: ' + str(round(pvalue, 2)))
            pvalue_list.append(pvalue)

            # save values for plots
            true_means.append(np.mean(true_sub_results))
            true_stds.append(np.std(true_sub_results))
            pred_means.append(np.mean(pred_sub_results))
            pred_stds.append(np.std(pred_sub_results))

        return pvalue_list, true_means, true_stds, pred_means, pred_stds, trial_id_list

    @staticmethod
    def show_each_pair(result_all_df):
        plt.figure()
        plt.plot(result_all_df['FPA_true'], result_all_df['FPA_tbme'], '.')
        plt.plot([-20, 60], [-20, 60], 'r-')
        plt.title('FPA_tbme')

        plt.figure()
        plt.plot(result_all_df['FPA_true'], result_all_df['FPA_estis'], '.')
        plt.plot([-20, 60], [-20, 60], 'r-')
        plt.title('FPA_esti')

    @staticmethod
    def compare_mean_error(result_all_df, type_name, x_tick_list=None, x_label=None):
        error_tbme = result_all_df['FPA_true'] - result_all_df['FPA_tbme']
        error_acc_ratio = result_all_df['FPA_true'] - result_all_df['FPA_estis']
        means_tbme, stds_tbme, id_list = ErrorBarFigure.get_mean_std(error_tbme, result_all_df[type_name])
        means_acc_ratio, stds_acc_ratio, _ = ErrorBarFigure.get_mean_std(error_acc_ratio, result_all_df[type_name])

        plt.figure(figsize=(14, 8))
        ErrorBarFigure.format_plot()
        bars, ebars = [], []
        for i_cate in range(len(id_list)):
            bars.append(plt.bar(i_cate, means_acc_ratio[i_cate], color='brown', width=0.4))
        ErrorBarFigure.draw_ebars(means_acc_ratio, stds_acc_ratio, id_list)

        x_locs = []
        for i_cate in range(len(id_list)):
            bars.append(plt.bar(i_cate + 0.4, means_tbme[i_cate], color='slategray', width=0.4))
            x_locs.append(i_cate + 0.4)
        ErrorBarFigure.draw_ebars(means_tbme, stds_tbme, id_list, x_locs=x_locs)
        ax = plt.gca()
        x_tick_loc = [i + 0.2 for i in range(len(id_list))]
        ax.tick_params(labelsize=FONT_DICT['fontsize'])
        ax.set_ylabel('Mean error (°)', fontdict=FONT_DICT)
        ax.set_xticks(x_tick_loc)
        if x_tick_list is None:
            x_tick_list = [str(int(the_id)) for the_id in id_list]
        if x_label is None:
            x_label = type_name
        ax.set_xticklabels(x_tick_list, fontdict=FONT_DICT)
        ax.set_xlabel(x_label, fontdict=FONT_DICT)

        plt.legend([bars[0], bars[-1]], FPA_NAME_LIST_HS[1:3])

    @staticmethod
    def compare_mean_value(result_all_df, type_name, x_tick_list=None, x_label=None):
        values_tbme = result_all_df['FPA_tbme']
        values_acc_ratio = result_all_df['FPA_estis']
        values_vicon = result_all_df['FPA_true']
        means_tbme, stds_tbme, id_list = ErrorBarFigure.get_mean_std(values_tbme, result_all_df[type_name])
        means_acc_ratio, stds_acc_ratio, _ = ErrorBarFigure.get_mean_std(values_acc_ratio, result_all_df[type_name])
        means_vicon, stds_vicon, _ = ErrorBarFigure.get_mean_std(values_vicon, result_all_df[type_name])

        plt.figure(figsize=(14, 8))
        ErrorBarFigure.format_plot()
        bars = []
        for i_cate in range(len(id_list)):
            bars.append(plt.bar(i_cate, means_vicon[i_cate], color='black', width=0.25))

        for i_cate in range(len(id_list)):
            bars.append(plt.bar(i_cate + 0.25, means_acc_ratio[i_cate], color='brown', width=0.25))

        for i_cate in range(len(id_list)):
            bars.append(plt.bar(i_cate + 0.5, means_tbme[i_cate], color='slategray', width=0.25))

        ax = plt.gca()
        x_tick_loc = [i + 0.25 for i in range(len(id_list))]
        ax.tick_params(labelsize=FONT_DICT['fontsize'])
        ax.set_ylabel('Mean error (°)', fontdict=FONT_DICT)
        ax.set_xticks(x_tick_loc)
        if x_tick_list is None:
            x_tick_list = [str(int(the_id)) for the_id in id_list]
        if x_label is None:
            x_label = type_name
        ax.set_xticklabels(x_tick_list, fontdict=FONT_DICT)
        ax.set_xlabel(x_label, fontdict=FONT_DICT)
        plt.legend([bars[0], bars[len(bars)//2], bars[-1]], FPA_NAME_LIST_HS)

    @staticmethod
    def compare_mean_error_of_mean_sub_error(result_all_df, type_name, x_tick_list=None, x_label=None):
        """The mean error of each subject's mean error is used, which is the same as that of TBME"""
        sub_name_list = list(set(result_all_df['sub_name'].values))
        trial_id_list = list(set(result_all_df['trial_id'].values))
        # type_id_of_errors is used to distinguish error of different trials
        type_id_of_errors, error_of_subs_acc_ratio, error_of_subs_tbme = [], [], []
        for trial_id in trial_id_list:
            trial_result_df = result_all_df[result_all_df['trial_id'] == trial_id]

            for sub_name in sub_name_list:
                sub_df = trial_result_df[trial_result_df['sub_name'] == sub_name]
                sub_mean_error_acc_ratio = np.mean(sub_df['FPA_true'] - sub_df['FPA_estis'])
                error_of_subs_acc_ratio.append(sub_mean_error_acc_ratio)
                sub_mean_error_tbme = np.mean(sub_df['FPA_true'] - sub_df['FPA_tbme'])
                error_of_subs_tbme.append(sub_mean_error_tbme)
                type_id_of_errors.append(trial_id)
        type_id_of_errors_df, error_of_subs_acc_ratio_df, error_of_subs_tbme_df = \
            pd.Series(type_id_of_errors), pd.Series(error_of_subs_acc_ratio), pd.Series(error_of_subs_tbme)
        means_acc_ratio, stds_acc_ratio, _ = ErrorBarFigure.get_mean_std(error_of_subs_acc_ratio_df, type_id_of_errors_df)
        means_tbme, stds_tbme, id_list = ErrorBarFigure.get_mean_std(error_of_subs_tbme_df, type_id_of_errors_df)

        plt.figure(figsize=(14, 8))
        ErrorBarFigure.format_plot()
        bars, ebars = [], []
        for i_cate in range(len(id_list)):
            bars.append(plt.bar(i_cate, means_acc_ratio[i_cate], color='brown', width=0.4))
        ErrorBarFigure.draw_ebars(means_acc_ratio, stds_acc_ratio, id_list)

        x_locs = []
        for i_cate in range(len(id_list)):
            bars.append(plt.bar(i_cate + 0.4, means_tbme[i_cate], color='slategray', width=0.4))
            x_locs.append(i_cate + 0.4)
        ErrorBarFigure.draw_ebars(means_tbme, stds_tbme, id_list, x_locs=x_locs)
        ax = plt.gca()
        x_tick_loc = [i + 0.2 for i in range(len(id_list))]
        ax.tick_params(labelsize=FONT_DICT['fontsize'])
        ax.set_ylabel('Mean error (°)', fontdict=FONT_DICT)
        ax.set_xticks(x_tick_loc)
        if x_tick_list is None:
            x_tick_list = [str(int(the_id)) for the_id in id_list]
        if x_label is None:
            x_label = type_name
        ax.set_xticklabels(x_tick_list, fontdict=FONT_DICT)
        ax.set_xlabel(x_label, fontdict=FONT_DICT)
        plt.legend([bars[0], bars[-1]], FPA_NAME_LIST_HS[1:])

    @staticmethod
    def draw_ebars(means, stds, cate_id_list, x_locs=None):
        if x_locs == None:
            x_locs = range(len(cate_id_list))
        ebar, caplines, barlinecols = plt.errorbar(x_locs, means, stds,
                                                   capsize=0, ecolor='black', fmt='none', lolims=True, uplims=True,
                                                   elinewidth=LINE_WIDTH)
        for i_cap in range(2):
            caplines[i_cap].set_marker('_')
            caplines[i_cap].set_markersize(14)
            caplines[i_cap].set_markeredgewidth(LINE_WIDTH)

    @staticmethod
    def draw_half_ebars(means, stds, x_locs, bar_direction_bool):
        lolims, uplims = bar_direction_bool, bool(1-bar_direction_bool)
        ebar, caplines, barlinecols = plt.errorbar(x_locs, means, stds,
                                                   capsize=0, ecolor='black', fmt='none', lolims=lolims, uplims=uplims,
                                                   elinewidth=LINE_WIDTH)
        caplines[0].set_marker('_')
        caplines[0].set_markersize(14)
        caplines[0].set_markeredgewidth(LINE_WIDTH)

    @staticmethod
    def draw_error_bar_figure_trials(result_df):
        cate_name = 'trial_id'
        mean_result_df, cate_id_list = ErrorBarFigure.get_mean_result_df(result_df, cate_name)
        diff_values = mean_result_df['FPA true'] - mean_result_df['FPA esti']
        means, stds, _ = ErrorBarFigure.get_mean_std(diff_values, mean_result_df[cate_name])

        plt.figure(figsize=(6, 6))
        ErrorBarFigure.format_plot()
        bars = []
        for i_cate in range(len(cate_id_list)):
            bars.append(plt.bar(i_cate, means[i_cate], color='gray', width=0.7))

        plt.plot([-1, 3], [0, 0], linewidth=LINE_WIDTH, color='black')

        ErrorBarFigure.draw_ebars(means, stds, cate_id_list)
        ErrorBarFigure.set_fpa_errorbar_ticks()
        plt.savefig('fpa_figures/fpa error of speeds.png')

    @staticmethod
    def set_fpa_errorbar_ticks():
        ax = plt.gca()
        ax.set_xlim(-0.5, 2.5)
        ax.set_xticks(np.arange(0, 3, 1))
        ax.set_xticklabels(['1.0 m/s', '1.2 m/s', '1.4 m/s'], fontdict=FONT_DICT_SMALL)

        ax.set_ylim(-2.5, 2.5)
        y_range = range(-2, 3, 1)
        ax.set_yticks(y_range)
        ax.set_yticklabels(y_range, fontdict=FONT_DICT_SMALL)
        ax.set_ylabel('Average FPA error (deg)', labelpad=10, fontdict=FONT_DICT_SMALL)

    @staticmethod
    def draw_true_esti_compare_figure(result_df):
        cate_ids = result_df['subtrial_id']
        means_esti, stds_esti, cate_id_list = ErrorBarFigure.get_mean_std(result_df['FPA esti'], cate_ids)
        means_true, stds_true, cate_id_list = ErrorBarFigure.get_mean_std(result_df['FPA true'], cate_ids)

        plt.figure(figsize=(9, 6))
        ErrorBarFigure.format_plot()
        bars_esti, bars_true = [], []
        for cate_id in cate_id_list:
            bars_esti.append(plt.bar(cate_id, means_esti[cate_id], color='gray', width=0.38))
            bars_true.append(plt.bar(cate_id + 0.4, means_true[cate_id], color='black', width=0.38))

        legend_names = ['FPA foot IMU', 'FPA vicon']
        plt.legend([bars_esti[0], bars_true[0]], legend_names, fontsize=FONT_SIZE, frameon=False,
                   bbox_to_anchor=(0.4, 0.95))
        ErrorBarFigure.set_fpa_compare_ticks()
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        plt.savefig('fpa_figures/Average fpa.png')

    @staticmethod
    def set_fpa_compare_ticks():
        ax = plt.gca()
        ax.set_xticks(np.arange(0.19, 5, 1))
        ax.set_xticklabels(['B/L - 10', 'B/L - 5', 'B/L', 'B/L + 15', 'B/L + 30'], fontdict=FONT_DICT_SMALL)
        y_range = range(0, 36, 7)
        ax.set_yticks(y_range)
        ax.set_yticklabels(y_range, fontdict=FONT_DICT_SMALL)
        ax.set_ylabel('Average FPA (deg)', labelpad=10, fontdict=FONT_DICT_SMALL)

    @staticmethod
    def get_mean_std(values, cate_ids):
        means, stds = [], []
        cate_id_list = list(set(cate_ids))
        cate_id_list.sort()
        for cate_id in cate_id_list:
            cate_values = values[cate_ids == cate_id]
            means.append(np.mean(cate_values))
            stds.append(np.std(cate_values))
        return means, stds, cate_id_list

    @staticmethod
    def get_mean_result_df(result_df, cate_2_name):
        """
        Get the mean result of one subject
        :param result_df:
        :param cate_2_name:
        :return:
        """
        subject_id_list = list(set(result_df['subject_id']))
        subject_id_list.sort()
        cate_id_list = list(set(result_df[cate_2_name]))
        cate_id_list.sort()
        rows = []
        for subject_id in subject_id_list:
            for cate_id in cate_id_list:
                result_cate_df = result_df[
                    (result_df['subject_id'] == subject_id) & (result_df[cate_2_name] == cate_id)]
                rows.append(np.mean(result_cate_df))
        mean_result_df = pd.DataFrame(rows)
        mean_result_df.columns = result_df.columns
        return mean_result_df, cate_id_list
