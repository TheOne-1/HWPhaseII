import pandas as pd
import numpy as np
from PaperFigures import ErrorBarFigure
import matplotlib.pyplot as plt


detailed_result_df = pd.read_csv('detailed_result_df.csv', index_col=False)
ErrorBarFigure.draw_true_esti_compare_figure(detailed_result_df)
ErrorBarFigure.draw_error_bar_figure_trials(detailed_result_df)
ErrorBarFigure.draw_error_bar_figure_subtrials(detailed_result_df)
plt.show()






