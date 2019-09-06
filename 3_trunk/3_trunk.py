from const import TRUNK_TRIALS, TRIAL_NAMES
from ProcessorTrunk import ProcessorTrunk


# define the output parameter here
output_parameter_name = 'trunk_ml_angle'        # trunk_ap_angle or trunk_ml_angle
# define train set subject and trials here
train = {}      # train dict is empty because you want to use a white box method
# define test set subject and trials here
# test = {'190810LiuSensen': TRUNK_TRIALS, '190815WangHan': TRUNK_TRIALS}
test = {'190810LiuSensen': [TRIAL_NAMES[2]]}

trunk_processor = ProcessorTrunk(train, test, 200, 'l', output_parameter_name, 'trunk', data_type=0,
                                 do_input_norm=False, do_output_norm=False)
# trunk_processor.draw_subtrial_output_error_bar(trial_id=4)
trunk_processor.prepare_train_test()
trunk_processor.white_box_solution()











