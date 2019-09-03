from const import WALKING_TRIALS
from ProcessorFPA import ProcessorFPA

train = {}
# test = {'190810LiuSensen': WALKING_TRIALS}
test = {'190810LiuSensen': WALKING_TRIALS}

FPA_processor = ProcessorFPA(train, test, 200, 'l', 'FPA', 'l_foot', data_type=0,
                             do_input_norm=False, do_output_norm=False)
FPA_processor.prepare_train_test(trial_ids=[3, 6, 9])
FPA_processor.white_box_solution(200)
