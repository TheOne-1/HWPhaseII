from const import TRUNK_TRIALS, SUB_NAMES
from ProcessorTrunk import ProcessorTrunk


# define the output parameter here
output_parameter_name = 'trunk_ml_angle'        # trunk_ap_angle or trunk_ml_angle
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


test = {'190803LiJiayi': TRUNK_TRIALS, '190806SunDongxiao': TRUNK_TRIALS, '190806WangDianxin': TRUNK_TRIALS,
                  '190810LiuSensen': TRUNK_TRIALS, '190815WangHan': TRUNK_TRIALS, '190815QiuYue': TRUNK_TRIALS,
                  '190816YangCan': TRUNK_TRIALS, '190820FuZhenzhen': TRUNK_TRIALS, '190820FuZhinan': TRUNK_TRIALS,
                  '190822HeMing': TRUNK_TRIALS, '190828LiangJie': TRUNK_TRIALS,
                  '190829JiBin': TRUNK_TRIALS, '190829ZhaoJiamin': TRUNK_TRIALS, '190831XieJie': TRUNK_TRIALS,
                  '190824ZhangYaqian': TRUNK_TRIALS, '190831GongChangyang': TRUNK_TRIALS, '190813ZengJia': TRUNK_TRIALS,
                  '190813Caolinfeng': TRUNK_TRIALS}

test = {'190829JiBin': TRUNK_TRIALS}

trunk_processor = ProcessorTrunk(train, test, 200, 'l', output_parameter_name, 'trunk', data_type=0,
                                 do_input_norm=False, do_output_norm=False, show_plots=False)

subtrials = [0, 2, 4]


trunk_processor.comp_filter_mix_ratio_ap = .007
trunk_processor.comp_filter_mix_ratio_ml = .01
trunk_processor.filter_cutoff_freq = 1


if output_parameter_name == 'trunk_ml_angle':
    print(trunk_processor.comp_filter_mix_ratio_ml)
elif output_parameter_name == 'trunk_ap_angle':
    print(trunk_processor.comp_filter_mix_ratio_ap)

for index, subject in enumerate(test):
    trunk_processor.calibrate_subject(subject_name=subject)
    trunk_processor.prepare_train_test(subject_ids=[SUB_NAMES.index(subject)], subtrial_ids=subtrials)
    trunk_processor.white_box_solution()








