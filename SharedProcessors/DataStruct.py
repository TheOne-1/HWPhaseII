from const import TRIAL_NAMES, SUB_NAMES, DATA_COLUMNS_IMU
import numpy as np
import pandas as pd


class DataStruct:
    def __len__(self):
        return self.__sample_num

    def append(self, input_array, output_array, subject_id, trial_id, subtrial_array):
        # this method has to be overwritten
        raise NotImplementedError('this convert_step_input method has to be overwritten')

    def get_all_data(self, subject_ids=None, trial_ids=None, subtrial_ids=None):
        # this method has to be overwritten
        raise NotImplementedError('this convert_step_input method has to be overwritten')


class DataStructSample(DataStruct):
    def __init__(self, input_dim=6, output_dim=1):
        self.__input_dim = input_dim
        self.__output_dim = output_dim
        self.__sample_num = 0

        self.__col_input = ['input_' + str(num) for num in range(input_dim)]
        self.__col_output = ['output_' + str(num) for num in range(output_dim)]
        self.__col_param = ['subject_id', 'trial_id', 'subtrial_id']
        self.__col_all = self.__col_input + self.__col_output + self.__col_param

        self.__data_dim = len(self.__col_all)

        self.__data_df = pd.DataFrame(columns=self.__col_all)
        # self.__mask_array = np.zeros([0, 1])    # used to mask bad samples, which is more efficient then delete
        # self.__input_array, self.__output_array = np.zeros([0, input_dim]), np.zeros(0, output_dim)

    def append(self, input_array, output_array, subject_id, trial_id, subtrial_array):
        if len(list({input_array.shape[0], output_array.shape[0], subtrial_array.shape[0]})) != 1:
            raise AssertionError('the length of input_array, output_array, and param_array needs to be the same')
        if input_array.shape[1] != self.__input_dim:
            raise AssertionError('Wrong input shape')
        if len(output_array.shape) != 1:
            if output_array.shape[1] != self.__output_dim:
                raise AssertionError('Wrong output shape')

        self.__sample_num += input_array.shape[0]

        data_array = np.zeros([input_array.shape[0], self.__data_dim])
        data_array[:, :self.__input_dim] = input_array
        data_array[:, self.__input_dim:self.__input_dim+self.__output_dim] = output_array.reshape([-1, 1])
        data_array[:, -3] = subject_id
        data_array[:, -2] = trial_id
        data_array[:, -1] = subtrial_array        # subtrial_array should be a 1-d array
        new_data_df = pd.DataFrame(data_array, columns=self.__col_all)
        self.__data_df = self.__data_df.append(new_data_df)
        self.__data_df = self.__data_df.reset_index(drop=True)
        self.__sample_num = self.__data_df.shape[0]

    def get_all_data(self, subject_ids=None, trial_ids=None, subtrial_ids=None):
        """
        Get data by samples. If no criteria is provided, all the data will be returned.
        :param subject_ids: list
            The id of the target subjects.
        :param trial_ids: list
            The id of the target trials.
        :param subtrial_ids: list
            The id of the target subtrials.
        :return: input_array, output_array, _
        """
        data_df = self.__data_df
        if data_df.shape[0] == 0:
            return None, None, None     # return None if the data_df is empty
        if subject_ids is not None:
            data_df = data_df[data_df['subject_id'].isin(subject_ids)]
        if trial_ids is not None:
            data_df = data_df[data_df['trial_id'].isin(trial_ids)]
        if subtrial_ids is not None:
            data_df = data_df[data_df['subtrial_id'].isin(subtrial_ids)]
        input_array = data_df[self.__col_input].values
        output_array = data_df[self.__col_output].values
        id_df = data_df[['subject_id', 'trial_id', 'subtrial_id']]
        return input_array, output_array, id_df


class DataStructStep(DataStruct):
    """
    This is a simple data structure for pop or insert input, output, status all together
    """

    def __init__(self):
        self.__input_list, self.__aux_input_list, self.__output_list = [], [], []
        self.__sub_id_list, self.__trial_id_list, self.__subtrial_id_list = [], [], []
        self.__step_num = 0

    def __len__(self):
        return self.__step_num

    def append(self, inertial_input_list, aux_input_list, output_list, subject_name, trial_name):
        if len(inertial_input_list) != len(output_list) or len(aux_input_list) != len(output_list):
            raise ValueError('The length of input and output should be the same')
        self.__step_num += len(inertial_input_list)
        self.__input_list.extend(inertial_input_list)
        self.__aux_input_list.extend(aux_input_list)
        self.__output_list.extend(output_list)
        subject_id = SUB_NAMES.index(subject_name)
        trial_id = TRIAL_NAMES.index(trial_name)
        for i_item in range(len(inertial_input_list)):
            self.__sub_id_list.append(subject_id)
            self.__trial_id_list.append(trial_id)

    def pop(self, index):
        self.__input_list.pop(index)
        self.__aux_input_list.pop(index)
        self.__output_list.pop(index)
        self.__sub_id_list.pop(index)
        self.__trial_id_list.pop(index)
        self.__step_num -= 1

    def get_input_output_list(self):
        return self.__input_list, self.__aux_input_list, self.__output_list

    def get_sub_id_list(self):
        return self.__sub_id_list

    def get_trial_id_list(self):
        return self.__trial_id_list

    def get_all_data(self, subject_ids=None, trial_ids=None, subtrial_ids=None, strike_off_from_IMU=True):
        """
        Return a list of input and a list of output. The length of corresponding input and output might be
         different due to different strike-off detection method. Steps are from toe-off to toe-off.
        :param subject_ids: list
            The id of the target subjects.
        :param trial_ids: list
            The id of the target trials.
        :param subtrial_ids: list
            The id of the target subtrials.
        :param strike_off_from_IMU:
        :return:
        """
        input_list, output_list = [], []
        input_array, output_array, data_df = self.prepare_data_by_sample(subject_ids, trial_ids, subtrial_ids)


