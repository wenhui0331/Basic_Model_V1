from D_SplitData.EquiDistanceSplitModule.EquiDistanceSplitHandle import EquiDistanceSplitClass
import D_SplitData
import pandas as pd
import unittest
from unittest import TestCase
from unittest import mock
import warnings 
warnings.filterwarnings("ignore")

class TestEquiDistanceSplit(TestCase):

    # 导入测试数据

    def test_equi_distance_split(self):

        data = pd.read_csv('D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/sxk_person_model_sample20210407.csv')
        data1 = data[['rela_bank_mon_num','label']]
        data1 = data1.fillna(-9999999)
        (SplitedPoint,splitedRegroup,IV,WOEDict) = D_SplitData.EquiDistanceSplitClass(20,-9999999,'label',1).equi_distance_split(data1,'rela_bank_mon_num')
        print(SplitedPoint)
        print(splitedRegroup)
        print(IV)
        print(WOEDict)
        return SplitedPoint

if __name__ == '__main__':  
    unittest.main()




