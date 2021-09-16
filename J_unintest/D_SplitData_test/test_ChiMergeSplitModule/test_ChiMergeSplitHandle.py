from D_SplitData.ChiMergeSplitModule.ChiMergeSplitHandle import ChiMergeSplitClass
from D_SplitData.EquiFrequentSplitModule.EquiFrequentSplitHandle import EquiFrequentSplitClass
from D_SplitData.SplitPointApplyModule.SplitPointApplyHandle import SplitPointApplyClass
import pandas as pd
import numpy as np
import unittest
from unittest import TestCase
from unittest import mock
import warnings 
warnings.filterwarnings("ignore")
from time import *



class TestChiMergeSplitClass(TestCase):

    # 导入测试数据
    def test_chi(self):
        data = pd.DataFrame(
            np.array([[1,2,100,200],
                [3,4,40,500],
                [100,560,345,980]])
                )

        a = ChiMergeSplitClass(-9999999,'label',0.05,0.95,5,1,50).chi(data)
        print(a)
        return a
    
    def test_core_chi(self):


        # 输出等频分箱结果
        data = pd.read_csv('D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/sxk_person_model_sample20210407.csv')
        data1 = data[['loan_pn_guan_num_rate','label']]
        data1 = data1.fillna(-9999999)
        SplitedPoint = EquiFrequentSplitClass(50,-9999999,'label',1).equi_freuent_split(data1,'loan_pn_guan_num_rate')
        #[0.0, 0.0833, 0.1429, 0.1667, 0.2, 0.25, 0.2857, 0.3333, 0.4, 0.5, 0.5714, 0.6, 0.6667, 0.75, 0.8, 0.875]
        (SplitedPoint, splitedBG) =ChiMergeSplitClass(-9999999,'label',0.05,0.95,5,1,50).core_chi(data1,'loan_pn_guan_num_rate',SplitedPoint,10)
        print(SplitedPoint)
        print(splitedBG)
        splitedBG.to_excel('C:/Users/Administrator/Desktop/1.xlsx')
        return splitedBG
#python -m unittest J_unintest/D_SplitData_test/test_ChiMergeSplitModule/test_ChiMergeSplitHandle.py

    def test_single_bin_pcnt(self):

        # 测试数据

        data = pd.read_csv('D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/sxk_person_model_sample20210407.csv')
        colForSplit = 'cred_total_amt'#'cred_biz_bank_num'
        data1 = data[[colForSplit,'label']]
        data1 = data1.fillna(-9999999)
        colForSplit = 'cred_total_amt'
        SplitedPoint = [100000,200000,300000,500000]
        (SplitedPoint,splitedBG) = ChiMergeSplitClass(-9999999,'label',0.05,0.95,5,1,50).single_bin_pcnt(data1,colForSplit,SplitedPoint)
        print(splitedBG)
        print(SplitedPoint)
        return (SplitedPoint,splitedBG)

    def test_Chi_Merge_Cut(self):

        # 测试数据
        data = pd.read_csv('D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/sxk_person_model_sample20210407.csv')

        # 测试调用
        #late_year_enqu_count\rela_bank_mon_num\late_year_enqu_count\late_year_guan_count\zc_asset_balan\ago_3_mon_xy_guan_num
        col = 'ago_6_mon_max_od_amt'	
        
        data1 = data[[col,'label']]
        data1 = data1.fillna(-9999999)
        begin_time = time()
        (SplitedPoint,splitedBG) = ChiMergeSplitClass(-9999999,'label',0.05,0.95,5,1,50).chi_merge_split(data1,col)
        end_time = time()
        run_time = end_time - begin_time
        print('卡方分箱所花的时间',run_time)
        print('卡方分箱所花的时间',run_time)
        print('卡方分箱所花的时间',run_time)
        print('卡方分箱所花的时间',run_time)
        # python -m unittest J_unintest/D_SplitData_test/test_ChiMergeSplitModule/test_ChiMergeSplitHandle.py
        # 分配切点
        data1[col+'_Bin'] = SplitPointApplyClass(-9999999,'label').split_Bin(data1, col, SplitedPoint)  
        
        (splitResult1,IV,WOEDict) = SplitPointApplyClass(-9999999,'label').split_result(data1, col+'_Bin', SplitedPoint) 
        ### 分箱描述性分析
        print(splitResult1)
        print(IV)
        print(WOEDict)
        return (splitedBG,SplitedPoint)


    def test_one_step_chimerge(self):

        # 测试数据
        data = pd.read_csv('D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/sxk_person_model_sample20210407.csv')

        data1 = data[['cl_prod_enqu_reco_rate','label']]

        data1 = data1.fillna(-99999)

        colForSplit = 'cl_prod_enqu_reco_rate'
        SplitedPoint = [-1.0, 0.6667]

        (SplitedPoint, splitedBG) = ChiMergeSplitClass(-9999999,'label',0.05,0.95,5,1,50).one_step_chimerge(data1,colForSplit,SplitedPoint)

        # (SplitedPoint, splitedBG) = ChiMergeSplitClass(-9999999,'label',0.05,0.95,5,1,50).chi_merge_split(data1,colForSplit)

        print(splitedBG)

        return (SplitedPoint, splitedBG)



if __name__ == '__main__':  
    unittest.main()

