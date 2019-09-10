from Initializer import SubjectDataInitializer
from const import RAW_DATA_PATH, PROCESSED_DATA_PATH, SUB_NAMES, SUB_AND_TRIALS

for subject_folder in SUB_NAMES[15:16]:
    trials = SUB_AND_TRIALS[subject_folder]
    readme_xls = RAW_DATA_PATH + subject_folder + '\\readme\\readme_' + subject_folder + '.xlsx'
    my_initializer = SubjectDataInitializer(PROCESSED_DATA_PATH, subject_folder, trials, readme_xls,
                                            initialize_100Hz=False, initialize_200Hz=True, initialize_1000Hz=True,
                                            check_sync=False, check_walking_period=False)
