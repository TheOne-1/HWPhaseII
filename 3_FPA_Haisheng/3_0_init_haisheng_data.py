import matplotlib.pyplot as plt
from const import SUB_NAMES_HS
from DataProcessorHS import DataInitializerHS, ParamInitializerHS
from ProcessorFPAHS import InitFPA


for sub_folder in SUB_NAMES_HS[2:]:
    print(sub_folder)

    # data_initializer = DataInitializerHS(sub_folder)
    # data_initializer.start_init()
    param_initializer = ParamInitializerHS(sub_folder)
    param_initializer.start_init()

    fpa_initializer = InitFPA(sub_folder)
    fpa_initializer.backward_fpa_estimation()

    plt.show()
















