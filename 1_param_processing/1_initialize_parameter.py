from const import PROCESSED_DATA_PATH, RAW_DATA_PATH, SUB_NAMES, SUB_AND_WALKING_TRIALS
from ParameterProcessor import ParamProcessor, TrunkStaticProcessor

for subject_folder in SUB_NAMES:
    readme_xls = RAW_DATA_PATH + subject_folder + '\\readme\\readme_' + subject_folder + '.xlsx'

    init100 = False
    init200 = True
    plot = False

    my_static_trunk_init = TrunkStaticProcessor(
        subject_folder, readme_xls, plot_strike_off=plot,
        initialize_100Hz=init100, initialize_200Hz=init200)
    my_static_trunk_init.start_initalization(PROCESSED_DATA_PATH)

    trials = SUB_AND_WALKING_TRIALS[subject_folder]
    my_initializer = ParamProcessor(
        subject_folder, readme_xls, trials, plot_strike_off=plot,
        initialize_100Hz=init100, initialize_200Hz=init200)
    my_initializer.start_initalization(PROCESSED_DATA_PATH)
