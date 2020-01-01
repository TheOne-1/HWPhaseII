import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from const import LINE_WIDTH, FONT_DICT_SMALL, FONT_SIZE, FPA_NAME_LIST, SUB_NAMES, FONT_DICT
from Evaluation import Evaluation


class PaperFigure:
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
    def each_sub_fig(result_all_df):
        sub_ids = list(set(result_all_df['subject_id'].values))
        sub_ids.sort()
        predict_result_df = pd.DataFrame()
        for sub_id in sub_ids:
            sub_result_df = result_all_df[result_all_df['subject_id'] == sub_id]
            pearson_coeff, RMSE, mean_error = Evaluation.plot_fpa_result(
                sub_result_df[FPA_NAME_LIST[0]], sub_result_df[FPA_NAME_LIST[2]], int(sub_id))
            # plt.figure()
            # plt.plot(sub_result_df[FPA_NAME_LIST[0]], sub_result_df[FPA_NAME_LIST[2]], 'g.')
            # plt.plot([-20, 60], [-20, 60], 'black')
            # plt.title(SUB_NAMES[int(sub_id)])
            predict_result_df = Evaluation.insert_prediction_result(
                predict_result_df, SUB_NAMES[int(sub_id)], pearson_coeff, RMSE, mean_error)
        Evaluation.export_prediction_result(predict_result_df)


class ErrorBarFigure(PaperFigure):

    @staticmethod
    def show_each_pair(result_all_df):
        plt.figure()
        plt.plot(result_all_df[FPA_NAME_LIST[0]], result_all_df[FPA_NAME_LIST[1]], '.')
        plt.plot([-20, 60], [-20, 60], 'r-')
        plt.title(FPA_NAME_LIST[1])

        plt.figure()
        plt.plot(result_all_df[FPA_NAME_LIST[0]], result_all_df[FPA_NAME_LIST[2]], '.')
        plt.plot([-20, 60], [-20, 60], 'r-')
        plt.title(FPA_NAME_LIST[2])

    @staticmethod
    def compare_mean_error(result_all_df, type_name, x_tick_list=None, x_label=None):
        error_tbme = result_all_df[FPA_NAME_LIST[0]] - result_all_df[FPA_NAME_LIST[1]]
        error_acc_ratio = result_all_df[FPA_NAME_LIST[0]] - result_all_df[FPA_NAME_LIST[2]]
        means_tbme, stds_tbme, id_list = ErrorBarFigure.get_mean_std(error_tbme, result_all_df[type_name])
        means_acc_ratio, stds_acc_ratio, _ = ErrorBarFigure.get_mean_std(error_acc_ratio, result_all_df[type_name])

        plt.figure(figsize=(14, 8))
        ErrorBarFigure.format_plot()
        bars, ebars = [], []
        for i_cate in range(len(id_list)):
            bars.append(plt.bar(i_cate, means_tbme[i_cate], color='slategray', width=0.4))
        ErrorBarFigure.draw_ebars(means_tbme, stds_tbme, id_list)

        x_locs = []
        for i_cate in range(len(id_list)):
            bars.append(plt.bar(i_cate + 0.4, means_acc_ratio[i_cate], color='brown', width=0.4))
            x_locs.append(i_cate + 0.4)
        ErrorBarFigure.draw_ebars(means_acc_ratio, stds_acc_ratio, id_list, x_locs=x_locs)
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

        plt.legend([bars[0], bars[-1]], FPA_NAME_LIST[1:3])

    @staticmethod
    def compare_mean_value(result_all_df, type_name, x_tick_list=None, x_label=None):
        values_tbme = result_all_df[FPA_NAME_LIST[1]]
        values_acc_ratio = result_all_df[FPA_NAME_LIST[2]]
        values_vicon = result_all_df[FPA_NAME_LIST[0]]
        means_tbme, stds_tbme, id_list = ErrorBarFigure.get_mean_std(values_tbme, result_all_df[type_name])
        means_acc_ratio, stds_acc_ratio, _ = ErrorBarFigure.get_mean_std(values_acc_ratio, result_all_df[type_name])
        means_vicon, stds_vicon, _ = ErrorBarFigure.get_mean_std(values_vicon, result_all_df[type_name])

        plt.figure(figsize=(14, 8))
        ErrorBarFigure.format_plot()
        bars = []
        for i_cate in range(len(id_list)):
            bars.append(plt.bar(i_cate, means_vicon[i_cate], color='black', width=0.25))

        for i_cate in range(len(id_list)):
            bars.append(plt.bar(i_cate + 0.25, means_tbme[i_cate], color='slategray', width=0.25))

        for i_cate in range(len(id_list)):
            bars.append(plt.bar(i_cate + 0.5, means_acc_ratio[i_cate], color='brown', width=0.25))

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
        plt.legend([bars[0], bars[len(bars)//2], bars[-1]], FPA_NAME_LIST)

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
    def draw_error_bar_figure_subtrials(result_df):
        cate_name = 'subtrial_id'
        mean_result_df, cate_id_list = ErrorBarFigure.get_mean_result_df(result_df, cate_name)
        diff_values = mean_result_df['FPA true'] - mean_result_df['FPA esti']
        means, stds, _ = ErrorBarFigure.get_mean_std(diff_values, mean_result_df[cate_name])

        plt.figure(figsize=(8, 6))
        ErrorBarFigure.format_plot()
        bars, ebars = [], []
        for i_cate in range(len(cate_id_list)):
            bars.append(plt.bar(i_cate, means[i_cate], color='gray', width=0.7))

        plt.plot([-1, 5], [0, 0], linewidth=LINE_WIDTH, color='black')
        ebar, caplines, barlinecols = plt.errorbar(range(len(cate_id_list)), means, stds,
                                                   capsize=0, ecolor='black', fmt='none', lolims=True, uplims=True,
                                                   elinewidth=LINE_WIDTH)
        for i_cap in range(2):
            caplines[i_cap].set_marker('_')
            caplines[i_cap].set_markersize(14)
            caplines[i_cap].set_markeredgewidth(LINE_WIDTH)
        ErrorBarFigure.set_fpa_errorbar_ticks_subtrial()
        plt.savefig('fpa_figures/fpa error of subtrials.png')

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
    def set_fpa_errorbar_ticks_subtrial():
        ax = plt.gca()
        ax.set_xlim(-0.5, 4.5)
        ax.set_xticks(np.arange(0, 5, 1))
        ax.set_xticklabels(['B/L - 10', 'B/L - 5', 'B/L', 'B/L + 15', 'B/L + 30'], fontdict=FONT_DICT_SMALL)

        ax.set_ylim(-4, 4)
        y_range = range(-4, 5, 2)
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