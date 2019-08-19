from Processor import Processor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ProcessorFPA(Processor):
    def convert_input_output(self, inputs, outputs, sampling_fre):
        # 等到step based写完再写
        pass

    def draw_subtrial_output_error_bar(self):
        _, _, data_df = self.train_data.get_all_data()
        subtrial_id_array = data_df['subtrial_id']
        subtrial_ids = list(set(subtrial_id_array))

        mean_list, std_list = [], []
        trial_df = data_df[data_df['trial_id'] == 9]
        for subtrial_id in subtrial_ids:
            subtrial_df = trial_df[trial_df['subtrial_id'] == subtrial_id]
            subtrial_outputs = subtrial_df['output_0']
            mean_list.append(np.mean(subtrial_outputs))
            std_list.append(np.std(subtrial_outputs))

        x_bar = [i_bar for i_bar in range(len(subtrial_ids))]
        plt.figure()
        plt.bar(x_bar, mean_list)
        plt.errorbar(x_bar, mean_list, yerr=std_list, fmt='none')
        plt.show()


