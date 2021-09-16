import pandas as pd
from F_SelectChar.FeaSelectProcessModule.FeaStepSelectWDHandle import WDFeaStepwiseSelectClass

data = pd.read_excel('D:/15. 建模包测试结果/逐步回归等函数测试.xlsx')

feat_weakcorr_ivsorted1 = ['QY_未结清欠息金额_Bin_WOE','QY_欠息总额_Bin_WOE',
'QY_已结清欠息笔数_Bin_WOE','QY_五级分类状态_Bin_WOE','QY_近4年已结清欠息金额_Bin_WOE',
'QY_已结清欠息金额_Bin_WOE','QY_未来6个月到期业务笔数_Bin_WOE',
'QY_信贷业务种类数_Bin_WOE','QY_信用业务笔数_Bin_WOE',
'QY_未来6个月到期业务总额_Bin_WOE','QY_贷款业务总发生额/未结清信用总发生额_Bin_WOE',
'QY_未结清承兑敞口_Bin_WOE', 'QY_对外担保五级分类为非正常的笔数_Bin_WOE',
'QY_对外担保五级分类为非正常的余额_Bin_WOE','QY_贷款业务总余额/未结清信用总余额_Bin_WOE',
'QY_未结清承兑笔数_Bin_WOE','QY_平均出资金额_Bin_WOE']


# 第一个函数测试
(feat_nocorr,pDropVar,summary) = WDFeaStepwiseSelectClass(0.05,'y').linear_regression_significance(data,feat_weakcorr_ivsorted1)

# 第二个函数测试
(feat_nocorr,StepWiseDropVar) = WDFeaStepwiseSelectClass(0.05,'y').forward_delete_pvalue(data[feat_nocorr],data['y'])
# 第三个函数测试
(feat_nocorr,coefDropVar,lr_coe) = WDFeaStepwiseSelectClass(0.05,'y').forward_delete_coef(data[feat_nocorr],data['y'])
print('置信度被删除的变量',pDropVar)
print('逐步回归被删除的变量',StepWiseDropVar)
print('符号被删除的变量',coefDropVar)
print('最后留下变量',feat_nocorr)
#python -m unittest J_unintest/F_SelectChar_test/test_FeaSelectProcessModule/test_FesStepWDHandle.py