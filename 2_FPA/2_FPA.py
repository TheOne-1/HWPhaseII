from const import WALKING_TRIALS
from ProcessorFPA import ProcessorFPA

train = {'190810LiuSensen': WALKING_TRIALS}
# test = {'190810LiuSensen': WALKING_TRIALS}
test = {}

my_FPA_processor = ProcessorFPA(train, test, 200, 'l', 'FPA', 'l_foot', data_type=0)
my_FPA_processor.draw_subtrial_output_error_bar()























