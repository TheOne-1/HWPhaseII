import numpy as np


class ResultReader:
    @staticmethod
    def get_steps(trial_result):
        steps = []
        step_flags = trial_result['step_flag'].values
        i_flag = 0
        while i_flag < len(step_flags):
            if step_flags[i_flag] == 1:
                step_start = i_flag
                while step_flags[i_flag] != 2:
                    i_flag += 1
                step_end = i_flag
                steps.append([step_start, step_end])
            i_flag += 1
        return steps

    @staticmethod
    def get_fpas(fpa_name_list, trial_result, steps, num_of_steps, steps_to_skip):
        """
        The only steps with only one non-zero FPA are returned
        :param fpa_name_list:
        :return:
        """
        fpa_raw_values = trial_result[fpa_name_list].values
        fpa_num = len(fpa_name_list)
        step_result_list = []
        abandoned_step_num = 0
        for step in steps[steps_to_skip: num_of_steps + steps_to_skip + 1]:
            step_clip = fpa_raw_values[step[0]:step[1]]
            valid_value_flag = np.where(step_clip != 0)
            if len(valid_value_flag[1]) != fpa_num:
                abandoned_step_num += 1
                continue
            for i_fpa in range(fpa_num):
                if i_fpa not in valid_value_flag[1]:
                    abandoned_step_num += 1
                    break
            fpa_index = valid_value_flag[1].argsort()
            valid_fpas_unsorted = step_clip[valid_value_flag]
            valid_fpas = valid_fpas_unsorted[fpa_index]
            one_row_result = np.concatenate([valid_fpas])
            step_result_list.append(one_row_result)

        print('{num} steps abandoned'.format(num=abandoned_step_num))
        trial_result = np.zeros([len(step_result_list), fpa_num])
        for i_step in range(len(step_result_list)):
            trial_result[i_step, :] = step_result_list[i_step]
        return trial_result


























