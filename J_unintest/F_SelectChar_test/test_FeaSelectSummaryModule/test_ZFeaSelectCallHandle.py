import pandas as pd
import unittest
from unittest import TestCase
from unittest import mock
from time import *
from F_SelectChar.FeaSelectSummaryModule.ZFeaSelectCallHandle import FeaSelectProcessClass   # 导入需要测试的单元     字典处理py文件
from Z2_TestFunction.FakeTestDataModule.FakeTestDataHandle import FakeTestDataClass


import json

class TestFeaSelectProcess(TestCase):

    ## 开始使用faker造数功能: 造数pandas数据，[certid,label,c1]
    ## 开始使用faker造数功能: 造数pandas数据，[certid,label,c1]
    ## 开始使用faker造数功能: 造数pandas数据，[certid,label,c1]


    numlength = 40000   # pandas的数据长度
    missrate = 0   # 缺失率造数

    begin_time = time()
    # 造身份证号码数据，作为pandas的唯一标识
    datafake = pd.DataFrame(FakeTestDataClass(numlength, missrate).fake_certid(numlength),columns = ['certid'])
    datafake = datafake.reset_index(level=0)
    del datafake['index']
    # 造好坏标签数据
    datafake.loc[0:10000,'label'] = list(FakeTestDataClass(10001, missrate).fake_bad01(0.01)[0])    # 控制不同的坏样本率是为了 造出合适的IV值
    datafake.loc[10000:20000,'label'] = list(FakeTestDataClass(10001, missrate).fake_bad01(0.01)[0])
    datafake.loc[20000:39999,'label'] = list(FakeTestDataClass(20000, missrate).fake_bad01(0.01)[0])
    num = 0
    for i in range(100):
        num = num + 1 
        tempdatafake1 = FakeTestDataClass(numlength, missrate).fake_scatter_and_nan([0.011,0.0222,0.367,0.436,0.589])  # 造类似分箱数据
        datafake.loc[:,'c'+str(num)] = list(tempdatafake1[0])

    train = datafake.loc[0:20000,:]  #测试集合
    test = datafake.loc[20000:40000,:] #训练集合
    end_time = time()
    run_time = end_time - begin_time
    print('faker造数所花的时间',run_time)
    print('faker造数所花的时间',run_time)
    print('faker造数所花的时间',run_time)
    print('faker造数所花的时间',run_time)
    psiThreshold = 0.0002   # 因为是随机造数 psi会小些
    target = 'label'
    del train['certid']
    del test['certid']

    pThreshold = 0.0001
    selection='stepwise'
    sle=0.05
    sls=0.05
    includes=[]

    # 特征筛选参数配置   # 因为是随机造数 所以整体上 阈值数据都会比较小
    ivThreshold = 0.01
    xgbSelectNum = None
    psiThreshold = 0.0002
    corrThreshold = 0.002
    pThreshold = 0.0001
    selection='stepwise'
    sle=0.05
    sls=0.05
    includes=[]
    vifThreshold = 1.001

    def test_feature_select(self):
        """
        功能描述：测试feature_select函数
        edited by 王文丹 2021/07/25
        
        """
        FSPC = FeaSelectProcessClass(self.ivThreshold,self.xgbSelectNum,self.psiThreshold,self.corrThreshold
					 ,self.vifThreshold,self.pThreshold,self.selection,self.sle,self.sls,includes=[])

        resDataAll = FSPC.feature_select(self.train,self.test,self.target,True,True)

        return resDataAll


#python -m unittest J_unintest/F_SelectChar_test/test_FeaSelectSummaryModule/test_ZFeaSelectCallHandle.py
    
    # def test_feature_select(self):

    #     # 导入json配置文件


    #     train = pd.read_csv('../code/train.csv') # 自己导入一个测试数据
    #     test= pd.read_csv('../code/test.csv')
    #     target='TARGET'

    #     # 调起数据清洗的类

    #     feaselectclass = FeaSelectProcessClass(xgbSelectNum=15  )


    #     resDataAll = feaselectclass.feature_select(train,test,target)

    #     return resDataAll



if __name__ == '__main__':  
    unittest.main()
