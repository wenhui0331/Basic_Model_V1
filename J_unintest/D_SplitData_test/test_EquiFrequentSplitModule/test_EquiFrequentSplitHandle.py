from D_SplitData.EquiFrequentSplitModule.EquiFrequentSplitHandle import EquiFrequentSplitClass
import pandas as pd
import unittest
from unittest import TestCase
from unittest import mock

class TestEquiDistanceSplit(TestCase):

    # 导入测试数据

    def test_equi_frequent_split(self):

        data = pd.read_csv('D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/sxk_person_model_sample20210407.csv')
        data1 = data[['zc_asset_balan','label']]
        data1 = data1.fillna(-9999999)
        SplitedPoint= EquiFrequentSplitClass(100,-9999999,'label',1).equi_freuent_split(data1,'zc_asset_balan')
        print(SplitedPoint)
        return SplitedPoint

if __name__ == '__main__':  
    unittest.main()
#python -m unittest J_unintest/D_SplitData_test/test_EquiFrequentSplitModule/test_EquiFrequentSplitHandle.py