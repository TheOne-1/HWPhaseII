from const import TRUNK_TRIALS, SUB_NAMES
from ProcessorTrunk import ProcessorTrunk


# define the output parameter here
output_parameter_name = 'trunk_ap_angle'        # trunk_ap_angle or trunk_ml_angle
# define train set subject and trials here
train = {}      # train dict is empty because you want to use a white box method
# define test set subject and trials here
# test = {'190803LiJiayi': TRUNK_TRIALS, '190806SunDongxiao': TRUNK_TRIALS, '190806WangDianxin': TRUNK_TRIALS,
#            '190810LiuSensen': TRUNK_TRIALS,  '190815QiuYue': TRUNK_TRIALS,
#            '190816YangCan': TRUNK_TRIALS, '190820FuZhenzhen': TRUNK_TRIALS, '190820FuZhinan': TRUNK_TRIALS,
#             '190822HeMing': TRUNK_TRIALS, '190826MouRongzi': TRUNK_TRIALS, '190828LiangJie': TRUNK_TRIALS,
#              '190831XieJie': TRUNK_TRIALS,'190829ZhaoJiamin': TRUNK_TRIALS, '190824ZhangYaqian': TRUNK_TRIALS,
#         '190831GongChangyang': TRUNK_TRIALS, '190813ZengJia': TRUNK_TRIALS,
#             '190813Caolinfeng': TRUNK_TRIALS}
#  '190826MouRongzi': TRUNK_TRIALS,

test = {'190803LiJiayi': TRUNK_TRIALS, '190806SunDongxiao': TRUNK_TRIALS, '190806WangDianxin': TRUNK_TRIALS,
                  '190810LiuSensen': TRUNK_TRIALS, '190815WangHan': TRUNK_TRIALS, '190815QiuYue': TRUNK_TRIALS,
                  '190816YangCan': TRUNK_TRIALS, '190820FuZhenzhen': TRUNK_TRIALS, '190820FuZhinan': TRUNK_TRIALS,
                  '190822HeMing': TRUNK_TRIALS, '190828LiangJie': TRUNK_TRIALS,
                  '190829JiBin': TRUNK_TRIALS, '190829ZhaoJiamin': TRUNK_TRIALS, '190831XieJie': TRUNK_TRIALS,
                  '190824ZhangYaqian': TRUNK_TRIALS, '190831GongChangyang': TRUNK_TRIALS, '190813ZengJia': TRUNK_TRIALS,
                  '190813Caolinfeng': TRUNK_TRIALS}



#test = {'190815WangHan': TRUNK_TRIALS}

#test = {'190826MouRongzi': TRUNK_TRIALS}

#test = {'190829JiBin': TRUNK_TRIALS}

trunk_processor = ProcessorTrunk(train, test, 200, 'l', output_parameter_name, 'trunk', data_type=0,
                                 do_input_norm=False, do_output_norm=False, show_plots=False)

#trunk_processor.draw_subtrial_output_error_bar(trial_id=4)

trunk_processor.paramA = .007
print(trunk_processor.paramA)
for subject in test:
    trunk_processor.calibrate_subject(subject_name=subject)
    trunk_processor.prepare_train_test(subject_ids=[SUB_NAMES.index(subject)], subtrial_ids=[0, 2, 4])
    trunk_processor.white_box_solution()

trunk_processor.paramA = .01
print(trunk_processor.paramA)
for subject in test:
    trunk_processor.calibrate_subject(subject_name=subject)
    trunk_processor.prepare_train_test(subject_ids=[SUB_NAMES.index(subject)], subtrial_ids=[0, 2, 4])
    trunk_processor.white_box_solution()

trunk_processor.paramA = .03
print(trunk_processor.paramA)

for subject in test:
    trunk_processor.calibrate_subject(subject_name=subject)
    trunk_processor.prepare_train_test(subject_ids=[SUB_NAMES.index(subject)], subtrial_ids=[0, 2, 4])
    trunk_processor.white_box_solution()

trunk_processor.paramA = .05
print(trunk_processor.paramA)

for subject in test:
    trunk_processor.calibrate_subject(subject_name=subject)
    trunk_processor.prepare_train_test(subject_ids=[SUB_NAMES.index(subject)], subtrial_ids=[0, 2, 4])
    trunk_processor.white_box_solution()

trunk_processor.paramA = .08
for subject in test:
    trunk_processor.calibrate_subject(subject_name=subject)
    trunk_processor.prepare_train_test(subject_ids=[SUB_NAMES.index(subject)], subtrial_ids=[0, 2, 4])
    trunk_processor.white_box_solution()

trunk_processor.paramA = .10
for subject in test:
    trunk_processor.calibrate_subject(subject_name=subject)
    trunk_processor.prepare_train_test(subject_ids=[SUB_NAMES.index(subject)], subtrial_ids=[0, 2, 4])
    trunk_processor.white_box_solution()














