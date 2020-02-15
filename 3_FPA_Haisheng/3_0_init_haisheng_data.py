import matplotlib.pyplot as plt
from const import SUB_NAMES_HS
from DataProcessorHS import DataInitializerHS, ParamInitializerHS
from ProcessorFPAHS import InitFPA

test_date = '0210'

# for sub_folder in SUB_NAMES_HS[:]:
#     data_initializer = DataInitializerHS(sub_folder)
#     data_initializer.start_init()
#     param_initializer = ParamInitializerHS(sub_folder)
#     param_initializer.start_init()
# plt.show()

fpa_initializer = InitFPA(test_date)
fpa_initializer.start_init()
# fpa_initializer.param_optimizer()

plt.show()
















