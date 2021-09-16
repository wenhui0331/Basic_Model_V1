import unittest
from unittest import TestCase
from unittest import mock
import pandas as pd
import numpy as np
from D_SplitData.SplitPointApplyModule.SplitPointApplyHandle import SplitPointApplyClass
from D_SplitData.EquiDistanceSplitModule.EquiDistanceSplitHandle import EquiDistanceSplitClass

class TestSplitPointApplyClass(TestCase):

    #导入测试数据

    def test_split_assign(self):

        data = pd.read_csv('D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/sxk_person_model_sample20210407.csv')
        data1 = data[['rela_bank_mon_num','label']]
        data1 = data1.fillna(-9999999)
        SplitedPoint = EquiDistanceSplitClass(numForSplit =20,specialValue=-9999999).EquiDistanceSplit(data1,'rela_bank_mon_num')
        print(SplitedPoint)

        data1['rela_mon_num_Bin'] = data1['rela_bank_mon_num'].apply(lambda x: SplitPointApplyClass(-9999999,'label').split_assign(x,SplitedPoint))

        print(data1)   # 检查分箱点的映射结果

        print(data1[data1['rela_mon_num_Bin'].isnull()].shape)   # 检查是否有没映射的点

        print(data1[data1['rela_mon_num_Bin']=='missing'].shape) # 检查 缺失值映射情况
        print(data[data['rela_bank_mon_num'].isnull()].shape) # 检查 缺失值映射情况
        print(data1[data1['rela_bank_mon_num']==-9999999].shape) # 检查 缺失值映射情况

        return (data1,SplitedPoint)

    def test_split_index(self):

        data = pd.read_csv('D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/sxk_person_model_sample20210407.csv')
        data1 = data[['rela_bank_mon_num','label']]
        data1 = data1.fillna(-9999999)
        SplitedPoint = EquiDistanceSplitClass(numForSplit =20,specialValue=-9999999).EquiDistanceSplit(data1,'rela_bank_mon_num')
        print(SplitedPoint)

        data1['rela_mon_num_Bin'] = data1['rela_bank_mon_num'].apply(lambda x: SplitPointApplyClass(-9999999,'label').split_assign(x,SplitedPoint))

        a = data1.groupby(['rela_mon_num_Bin'])['label'].agg([("count","sum")])
        print('未排序调整：')
        print(a)
        b = pd.DataFrame(a, index = SplitPointApplyClass(-9999999,'label').split_index(SplitedPoint))
        print('排序调整后：')
        print(b)

        return (a,b)

    def test_split_Bin(self):
        data = pd.read_csv('D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/sxk_person_model_sample20210407.csv')
        data1 = data[['rela_bank_mon_num','label']]
        data1 = data1.fillna(-9999999)
        SplitedPoint = EquiDistanceSplitClass(numForSplit =20,specialValue=-9999999).EquiDistanceSplit(data1,'rela_bank_mon_num')
        print(SplitedPoint)

        data1['rela_bank_mon_num_Bin'] = SplitPointApplyClass(-9999999,'label').split_Bin(data1, 'rela_bank_mon_num', SplitedPoint)

        print(data1)   # 检查分箱点的映射结果

        print(data1[data1['rela_bank_mon_num_Bin'].isnull()].shape)   # 检查是否有没映射的点
        print(data1[data1['rela_bank_mon_num_Bin']=='missing'].shape) # 检查 缺失值映射情况
        print(data[data['rela_bank_mon_num'].isnull()].shape) # 检查 缺失值映射情况
        print(data1[data1['rela_bank_mon_num']==-9999999].shape) # 检查 缺失值映射情况

        return (data1,SplitedPoint)

    def test_split_result(self):

        data = pd.read_csv('D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/sxk_person_model_sample20210407.csv')
        data1 = data[['rela_bank_mon_num','label']]
        data1 = data1.fillna(-9999999)
        SplitedPoint = EquiDistanceSplitClass(numForSplit =20,specialValue=-9999999).EquiDistanceSplit(data1,'rela_bank_mon_num')
        print(SplitedPoint)

        data1['rela_bank_mon_num_Bin'] = SplitPointApplyClass(-9999999,'label').split_Bin(data1, 'rela_bank_mon_num', SplitedPoint)
        print(data1)

        a = SplitPointApplyClass(-9999999,'label').split_result(data1, 'rela_bank_mon_num_Bin', SplitedPoint)
        print(a['regroup'])
        print(a['IV'])
        print(a['WOEDict'])

        return a
















