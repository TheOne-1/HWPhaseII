from const import FPA_TRIALS, TRIAL_NAMES
from ProcessorFPA import ProcessorFPA
import matplotlib.pyplot as plt



train = {}
test = {'190810LiuSensen': [TRIAL_NAMES[3]]}

FPA_processor = ProcessorFPA(train, test, 200, 'l', 'FPA_steps', 'l_foot', data_type=1,
                             do_input_norm=False, do_output_norm=False)
FPA_processor.prepare_train_test(trial_ids=[3, 6, 9])
FPA_processor.white_box_solution()
plt.show()
