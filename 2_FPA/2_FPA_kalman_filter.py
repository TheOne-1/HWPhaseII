from const import FPA_TRIALS, TRIAL_NAMES, SUB_AND_WALKING_TRIALS
from ProcessorFPA import ProcessorFPA
import matplotlib.pyplot as plt



train = {}
# test = {'190810LiuSensen': [TRIAL_NAMES[3]], '190806SunDongxiao': [TRIAL_NAMES[3]]}
test = {'190803LiJiayi': FPA_TRIALS, '190806SunDongxiao': FPA_TRIALS, '190806WangDianxin': FPA_TRIALS,
        '190810LiuSensen': FPA_TRIALS, '190815WangHan': FPA_TRIALS}


FPA_processor = ProcessorFPA(train, test, 200, 'l', 'FPA_steps', 'l_foot', data_type=1,
                             do_input_norm=False, do_output_norm=False)
FPA_processor.prepare_train_test(trial_ids=[3, 6, 9])
plt.show()
