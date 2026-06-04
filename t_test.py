import numpy as np
import scipy.stats
import math

from scipy.stats import ttest_ind_from_stats
from scipy import stats

#0-50
mean1= 1.05
std1 = math.sqrt(0.0005)
n1 = 50

#50-100
mean2 = 1.02
std2 = math.sqrt(0.0005)
n2 = 30

#100-150
mean3 = 1.0
std3 = math.sqrt(0.001334)
n3 = 30

t_stat12, p_value12 = ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var=False)
print(f"T-statistic12: {t_stat12}, P-value12: {p_value12}")
t_stat23, p_value23 = ttest_ind_from_stats(mean2, std2, n2, mean3, std3, n3, equal_var=False)
print(f"T-statistic23: {t_stat23}, P-value23: {p_value23}")

