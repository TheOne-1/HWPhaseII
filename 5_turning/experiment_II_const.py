DATA_PATH_TURNING = 'D:\Tian\Research\Projects\ML Project\data\\20200114FPA_ExperimentIIData\\200402TEST'
PLACEMENT_OFFSET = 0
SUB_AND_TRIAL_ID_TURNING = {
    '200407TanTian': {'baseline': [0, 1, 2, 3, 4], 'walking_start': [6, 7, 8, 9, 10], 'turning': [12, 13, 14, 15, 16]}
}
SUB_NAMES_TURNING = list(SUB_AND_TRIAL_ID_TURNING.keys())
TRIAL_NAMES_TURNING = list(SUB_AND_TRIAL_ID_TURNING[SUB_NAMES_TURNING[0]].keys())
SUB_TURNING_OCCURANCE = {
    '200407TanTian': [18, 19, 18, 19, 20]
}
STEP_RESULT_COLUMNS = ('sub_name', 'baseline_0', 'baseline_1', 'baseline_2', 'baseline_3', 'baseline_4',
                       'walking_start_0', 'walking_start_1', 'walking_start_2', 'walking_start_3', 'walking_start_4',
                       'turning_0', 'turning_1', 'turning_2', 'turning_3', 'turning_4')
BASELINE_TRIAl_NUM = 5
WALKING_START_TRIAl_NUM = 5
TURNING_TRIAl_NUM = 5
MAX_STEP_NUM = 42

BASELINE_START_STEP, BASELINE_END_STEP = -11, -1
WALKING_START_TOTAL_STEP = 9
BEFORE_TURNING_STEP, AFTER_TRUNING_STEP = 4, 9
