from const import WALKING_TRIALS
from ProcessorTrunk import ProcessorTrunk

# define train set subject and trials here
train = {}      # train dict is empty because you want to use a white box method
# define test set subject and trials here
test = {'190810LiuSensen': WALKING_TRIALS}

trunk_processor = ProcessorTrunk(train, test, 200, 'l', 'trunk_ap_angle', 'trunk', data_type=0,
                                 do_input_norm=False, do_output_norm=False)
# trunk_processor.draw_subtrial_output_error_bar(trial_id=4)
trunk_processor.prepare_train_test(trial_ids=[4, 7, 10])
trunk_processor.white_box_solution()











