from const import TRUNK_TRIALS, SUB_AND_PARAM_TRIALS, TRIAL_NAMES, SUB_AND_WALKING_TRIALS
from ProcessorTrunk import ProcessorTrunk


# define the output parameter here
output_parameter_name = 'r_FPA'        # trunk_ap_angle or trunk_ml_angle
train = {}      # train dict is empty because you want to use a white box method
test = {'190829JiBin': TRIAL_NAMES[2:]}

trunk_processor = ProcessorTrunk(train, SUB_AND_WALKING_TRIALS, 200, 'l', output_parameter_name, 'trunk', data_type=0,
                                 do_input_norm=False, do_output_norm=False)
trunk_processor.prepare_train_test()
trunk_processor.white_box_solution()











