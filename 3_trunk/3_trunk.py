from const import TRUNK_TRIALS, TRIAL_NAMES
from ProcessorTrunk import ProcessorTrunk


# define the output parameter here
output_parameter_name = 'trunk_ml_angle'        # trunk_ap_angle or trunk_ml_angle
train = {}      # train dict is empty because you want to use a white box method
test = {'190820FuZhinan': TRUNK_TRIALS}

trunk_processor = ProcessorTrunk(train, test, 200, 'l', output_parameter_name, 'trunk', data_type=0,
                                 do_input_norm=False, do_output_norm=False)
# trunk_processor.draw_subtrial_output_error_bar(trial_id=4)
trunk_processor.prepare_train_test()
trunk_processor.white_box_solution()











