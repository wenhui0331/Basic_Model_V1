## 1. 包的初始化

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, roc_auc_score
import seaborn as sns
from .ModelEffectHandle import ModelEffectClass
from . import ModelEffectHandle


## 2. 包功能性描述
'''
概述：评分卡评价指标计算

group_by_df：转换数据格式，变为指标每一分区下好样本的个数、坏样本的个数、ks值、lift 等
get_KS：计算KS
get_AUC：计算AUC
get_GINI：计算gini
get_Report：评分卡各项评价指标汇总报告
plot_KS：绘制KS曲线，并保存图片文件
plot_LIFT：绘制lift曲线，并保存图片文件
plot_PR：绘制PR曲线，并保存图片文件
plot_ROC：绘制ROC曲线，并保存图片文件
plot_PR_F1：绘制召回率、精确率、f1-score曲线图，并保存图片文件
plot_KS_Distribution：绘制累计好坏客户的曲线图，和分数分布曲线，并保存图片文件
plot_Good_Bad_Ind：绘制好客户、坏客户、中间客户分数分布图，并保存图片文件
plot_Cumulative_Approve：绘制累计好坏通过率图，并保存图片文件



'''


## 3. 维护记录
'''
created by 何岸  2021/07/06


'''