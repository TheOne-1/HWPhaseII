import matplotlib.pyplot as plt
from TurningProcessor import TurningProcessor

test_date = '0410'

fpa_initializer = TurningProcessor(test_date)

# fpa_esti_sub_dict, fpa_tbme_sub_dict = fpa_initializer.get_trial_fpa(show_trial=True)
# fpa_initializer.save_results(fpa_esti_sub_dict, 'esti')
# fpa_initializer.save_results(fpa_tbme_sub_dict, 'tbme')

walking_start_array_esti, turning_array_esti = fpa_initializer.analyze_result('esti')
walking_start_array_tbme, turning_array_tbme = fpa_initializer.analyze_result('tbme')
fpa_initializer.draw_walking_start(walking_start_array_esti, walking_start_array_tbme)
fpa_initializer.draw_turning(turning_array_esti, turning_array_tbme)

plt.show()
