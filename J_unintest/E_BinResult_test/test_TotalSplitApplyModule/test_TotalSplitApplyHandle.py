from E_BinResult.TotalSplitApplyModule.TotalSplitApplyHandle import TotalSplitApplyClass
import pandas as pd
import numpy as np
import unittest
from unittest import TestCase
from unittest import mock
import warnings 
warnings.filterwarnings("ignore")
from time import *


class TestTotalSplitApplyClass(TestCase):

    def test_manual_adjustment_splitpoint(self):
        
        # 设置测试数据

        specialValue = -999999
        ylabel = 'label'
        cutParameters = {'最小单箱占比':0.05,'卡方值置信度':0.95, '卡方值自由度':4, '是否单调':1, '初始分箱数':50}
        needModifyDict = {'rela_bank_mon_num': [90.0, 153.0]}
        binMethod = 'chiCutMethod'

        oldBinSplitDict = {'芝麻开门':[1,2,3],'西瓜开门':[100,200,300],'草莓开门':[345,765,890]}
        needModifyDict1 = {'芝麻开门':[235,346,678],'草莓开门':[1,2,3]}

        oldBinSplitDict1 = TotalSplitApplyClass(specialValue, ylabel,cutParameters,binMethod,needModifyDict).manual_adjustment_splitpoint(oldBinSplitDict, needModifyDict1)

        print(oldBinSplitDict1)
        return oldBinSplitDict1

    def test_split_all_apply(self):

        # 设置测试数据
        data1 = pd.read_csv('D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/sxk_person_model_sample20210407.csv')

        # 'cred_total_amt','cred_biz_bank_num','rela_bank_mon_num'
        dataForSplit = data1[['ago_3_mon_cred_card_used_num','label']]
        dataForSplit = dataForSplit.fillna(-999999)

        begin_time = time()
        allFeature = ['ago_3_mon_cred_card_used_num']
        numFeature = ['ago_3_mon_cred_card_used_num']
        binMethod = 'chiCutMethod'
        specialValue = -999999
        ylabel = 'label'
        cutParameters = {'最小单箱占比':0.05,'卡方值置信度':0.95, '卡方值自由度':4, '是否单调':1, '初始分箱数':50}
        needModifyDict = {}
        # needModifyDict = {}
        (ivSumSplitedSort,splitedPointDictSort,regroupSplitedDictSort,WOEBinDictSort) = TotalSplitApplyClass(specialValue, ylabel,cutParameters,binMethod,needModifyDict).split_all_apply(dataForSplit, allFeature)
        end_time = time()
        run_time = end_time - begin_time
        print('卡方分箱所花的时间',run_time)
        print('卡方分箱所花的时间',run_time)
        print('卡方分箱所花的时间',run_time)
        print('卡方分箱所花的时间',run_time)
        print(ivSumSplitedSort)
        print(splitedPointDictSort)
        print(regroupSplitedDictSort)
        print(WOEBinDictSort)
        return (ivSumSplitedSort,splitedPointDictSort,regroupSplitedDictSort,WOEBinDictSort)


# python -m unittest J_unintest/E_BinResult_test/test_TotalSplitApplyModule/test_TotalSplitApplyHandle.py

if __name__ == '__main__':  
    unittest.main()    
