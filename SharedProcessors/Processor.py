import matplotlib.pyplot as plt
from DataReader import DataReader
from keras.layers import *
import scipy.interpolate as interpo
from sklearn.preprocessing import MinMaxScaler


class Processor:
    def __init__(self, train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, grf_side, param_name,
                 IMU_location, data_type, do_input_norm=True, do_output_norm=False, show_plots=False):
        """

        :param data_type: int
            0 for sample based data, 1 for sample based data with strike off, 2 for step based data
        :return:
        """
        self.train_sub_and_trials = train_sub_and_trials
        self.test_sub_and_trials = test_sub_and_trials
        self.sensor_sampling_fre = sensor_sampling_fre
        self.grf_side = grf_side
        self.do_input_norm = do_input_norm
        self.do_output_norm = do_output_norm
        self.param_name = param_name
        self.IMU_location = IMU_location
        self.show_plots = show_plots
        self.cal_angle = 0
        self.train_obj = DataReader(train_sub_and_trials, param_name, sensor_sampling_fre, grf_side, IMU_location)
        if test_sub_and_trials is not None:
            self.test_obj = DataReader(test_sub_and_trials, param_name, sensor_sampling_fre, grf_side, IMU_location)

        if data_type == 0:
            self.train_data = self.train_obj.prepare_data_by_sample()
            self.test_data = self.test_obj.prepare_data_by_sample()
        elif data_type == 1:
            self.train_data = self.train_obj.prepare_data_by_with_strike_off()
            self.test_data = self.test_obj.prepare_data_by_with_strike_off()
        elif data_type == 2:
            self.train_data = self.train_obj.prepare_data_by_step()
            self.test_data = self.test_obj.prepare_data_by_step()
        else:
            raise ValueError('Wrong data_type value. It has to be 0 or 1.')

    def prepare_train_test(self, subject_ids=None, trial_ids=None, subtrial_ids=None):
        """

        :param subject_ids: list
            The ids of the target subjects.
        :param trial_ids: list
            The ids of the target trials.
        :param subtrial_ids: list
            The ids of the target subtrials.
        :return: input_array, output_array, _
        """
        inputs, outputs, train_id_df = self.train_data.get_all_data(subject_ids, trial_ids, subtrial_ids)
        self._x_train, self._y_train = self.convert_input_output(inputs, outputs, train_id_df, self.sensor_sampling_fre)
        inputs, outputs, test_id_df = self.test_data.get_all_data(subject_ids, trial_ids, subtrial_ids)
        self._x_test, self._y_test = self.convert_input_output(inputs, outputs, test_id_df, self.sensor_sampling_fre)
        # do input normalization
        if self.do_input_norm:
            self.norm_input()
        if self.do_output_norm:
            self.norm_output()

    # convert the input from list to ndarray
    def convert_input_output(self, inputs, outputs, id_df, sampling_fre):
        # this method has to be overwritten
        raise NotImplementedError('this convert_step_input method has to be overwritten')

    @staticmethod
    def resample_channel(data_array, resampled_len):
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(1, -1)
        data_len = data_array.shape[1]
        data_step = np.arange(0, data_len)
        resampled_step = np.linspace(0, data_len, resampled_len)
        tck, data_step = interpo.splprep(data_array, u=data_step, s=0)
        data_resampled = interpo.splev(resampled_step, tck, der=0)[0]
        return data_resampled

    def norm_input(self):
        channel_num = self._x_train.shape[2]
        # save input scalar parameter
        self.main_max_vals,  self.main_min_vals = [], []
        for i_channel in range(channel_num):
            max_val = np.max(self._x_train[:, :, i_channel]) * 0.99
            min_val = np.min(self._x_train[:, :, i_channel]) * 0.99
            self._x_train[:, :, i_channel] = (self._x_train[:, :, i_channel] - min_val) / (max_val - min_val)
            self._x_test[:, :, i_channel] = (self._x_test[:, :, i_channel] - min_val) / (max_val - min_val)
            self.main_max_vals.append(max_val)
            self.main_min_vals.append(min_val)

        if hasattr(self, '_x_train_aux'):
            # MinMaxScaler is more suitable because StandardScalar will make the input greatly differ from each other
            aux_input_scalar = MinMaxScaler()
            self._x_train_aux = aux_input_scalar.fit_transform(self._x_train_aux)
            self._x_test_aux = aux_input_scalar.transform(self._x_test_aux)
            self.aux_max_vals = aux_input_scalar.data_max_.tolist()
            self.aux_min_vals = aux_input_scalar.data_min_.tolist()

    def norm_output(self):
        self.output_minmax_scalar = MinMaxScaler(feature_range=(1, 3))
        self._y_train = self._y_train.reshape(-1, 1)
        self._y_train = self.output_minmax_scalar.fit_transform(self._y_train)
        self.result_max_vals = self.output_minmax_scalar.data_max_[0]
        self.result_min_vals = self.output_minmax_scalar.data_min_[0]

    def norm_output_reverse(self, output):
        output = output.reshape(-1, 1)
        output = self.output_minmax_scalar.inverse_transform(output)
        return output.reshape(-1,)

    def draw_subtrial_output_error_bar(self, trial_id, param_name=''):
        _, _, id_df = self.train_data.get_all_data()
        subtrial_id_array = id_df['subtrial_id']
        subtrial_ids = list(set(subtrial_id_array))

        mean_list, std_list = [], []
        trial_df = id_df[id_df['trial_id'] == trial_id]
        for subtrial_id in subtrial_ids:
            subtrial_df = trial_df[trial_df['subtrial_id'] == subtrial_id]
            subtrial_outputs = subtrial_df['output_0']
            mean_list.append(np.mean(subtrial_outputs))
            std_list.append(np.std(subtrial_outputs))

        x_bar = [i_bar for i_bar in range(len(subtrial_ids))]
        plt.figure()
        plt.bar(x_bar, mean_list)
        plt.errorbar(x_bar, mean_list, yerr=std_list, fmt='none')
        plt.xlabel('subtrial id')
        plt.ylabel(param_name + 'angle')
        plt.show()

    # def find_feature(self):
    #     train_all_data = DataReader(self.train_sub_and_trials, self.param_name, self.sensor_sampling_fre,
    #                                 self.grf_side, self.IMU_location)
    #     train_all_data_list = train_all_data.get_all_data()
    #     train_all_data_list = Processor.clean_all_data(train_all_data_list, self.sensor_sampling_fre)
    #     input_list, output_list = train_all_data_list.get_input_output_list()
    #     x_train, feature_names = self.convert_input_output(input_list, self.sensor_sampling_fre)
    #     y_train = Processor.convert_output(output_list)
    #     sub_id_list = train_all_data_list.get_sub_id_list()
    #     trial_id_list = train_all_data_list.get_trial_id_list()
    #     Processor.gait_phase_and_correlation(input_list, y_train, channels=range(6))
    #     Processor.draw_correlation(x_train, y_train, sub_id_list, SUB_NAMES, feature_names)
    #     Processor.draw_correlation(x_train, y_train, trial_id_list, TRIAL_NAMES, feature_names)
    #     plt.show()
    #
    # @staticmethod
    # def gait_phase_and_correlation(input_list, output_array, channels=range(6)):
    #     sample_num = len(input_list)
    #     resample_len = 100
    #     plt.figure()
    #     plot_list = []
    #     for i_channel in channels:
    #         input_array = np.zeros([sample_num, resample_len])
    #         for i_sample in range(sample_num):
    #             channel_data = input_list[i_sample][:, i_channel]
    #             channel_data_resampled = Processor.resample_channel(channel_data, resample_len)
    #             input_array[i_sample, :] = channel_data_resampled
    #         pear_correlations = np.zeros([resample_len])
    #         for phase in range(resample_len):
    #             pear_correlations[phase] = stats.pearsonr(input_array[:, phase], output_array)[0]
    #         channel_plot, = plt.plot(pear_correlations, color=COLORS[i_channel])
    #         plot_list.append(channel_plot)
    #     plt.xlabel('gait phase')
    #     plt.ylabel('correlation')
    #     plt.legend(plot_list, DATA_COLUMNS_XSENS[0:max(channels)+1])
    #     plt.grid()
    #     plt.show()

    # @staticmethod
    # def draw_correlation(input_array, output_array, category_id_list, category_names, feature_names):
    #     category_id_set = set(category_id_list)
    #     category_id_array = np.array(category_id_list)
    #     for i_feature in range(input_array.shape[1]):
    #         plt.figure()
    #         plt.title(feature_names[i_feature])
    #         plot_list, plot_names = [], []
    #         i_category = 0
    #         for category_id in category_id_set:
    #             category_name = category_names[category_id]
    #             if 'mini' in category_name:
    #                 plot_pattern = 'x'
    #             else:
    #                 plot_pattern = '.'
    #             plot_names.append(category_name)
    #             category_index = np.where(category_id_array == category_id)[0]
    #             category_plot, = plt.plot(input_array[category_index, i_feature], output_array[category_index],
    #                                       plot_pattern, color=COLORS[i_category])
    #             plot_list.append(category_plot)
    #             i_category += 1
    #         plt.legend(plot_list, plot_names)

    # @staticmethod
    # def clean_all_data(all_sub_data_struct, sensor_sampling_fre):
    #     i_step = 0
    #     input_list, output_list = all_sub_data_struct.get_input_output_list()
    #     min_time_between_strike_off = int(sensor_sampling_fre * 0.15)
    #     while i_step < len(all_sub_data_struct):
    #         # delete steps without a valid loading rate
    #         strikes = np.where(input_list[i_step][:, 6] == 1)[0]
    #         if np.max(output_list[i_step]) <= 0:
    #             all_sub_data_struct.pop(i_step)
    #
    #         # delete steps without a valid strike time
    #         elif len(strikes) != 1:
    #             all_sub_data_struct.pop(i_step)
    #
    #         # delete a step if the duration between strike and off is too short
    #         elif not min_time_between_strike_off < input_list[i_step].shape[0] - strikes[0]:
    #             all_sub_data_struct.pop(i_step)
    #
    #         else:
    #             # step number only increase when no pop happens
    #             i_step += 1
    #     return all_sub_data_struct




