from unittest import result
import pandas as pd
import numpy as np 
import unittest
from unittest import TestCase
from unittest import mock
from scipy import stats
from unittest.mock import patch
from time import *
from F_SelectChar.FeaSelectProcessModule.FeaDisSelectHandle import FeaDisSelectClass# 导入需要测试的单元 
from Z2_TestFunction.FakeTestDataModule.FakeTestDataHandle import FakeTestDataClass    
#python -m unittest J_unintest/F_SelectChar_test/test_FeaSelectProcessModule/test_FeaDisSelectHandle.py

class TestFeaDisSelectHandle(TestCase):
    
    """
    功能描述：缺失值处理测试类

    """
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
    for i in range(20):
        num = num + 1 
        tempdatafake1 = FakeTestDataClass(numlength, missrate).fake_float_and_nan(2, 0, True, 1, 5)  # 造类似分箱数据
        datafake.loc[:,'c'+str(num)] = list(tempdatafake1[0])
        num = num + 1 
        tempdatafake2 = FakeTestDataClass(numlength, missrate).fake_scatter_and_nan(['(-inf,0]','(0,1]','(1,2]','(2,3]','(3,4]','missing'])  # 造类似分箱数据
        datafake.loc[:,'c'+str(num)] = list(tempdatafake2[0])
    train = datafake.loc[0:20000,:]  #测试集合
    test = datafake.loc[20000:40000,:] #训练集合
    end_time = time()
    run_time = end_time - begin_time
    print('faker造数所花的时间',run_time)
    print('faker造数所花的时间',run_time)
    print('faker造数所花的时间',run_time)
    print('faker造数所花的时间',run_time)

    ivThreshold = 0.02
    xgbSelectNum = 13
    target = 'label'

    # print(datafake.head(5))

    def test_iv_xy(self):

        """
        功能描述： 测试iV值计算函数 

        管理记录：
        1. edited by 王文丹 2021/07/24
        
        """
        IV = FeaDisSelectClass(self.ivThreshold,self.xgbSelectNum).iv_xy(self.train['c1'], self.train['label'])

        print(IV)

        return IV
    
    def test_iv_claculate(self):

        """
        功能描述： 测试iv_claculate函数
        
        管理记录：
        1. edited by 王文丹 2021/07/24
        
        """
        ivList = FeaDisSelectClass(self.ivThreshold,self.xgbSelectNum).iv_claculate(self.train, 'label', X=['c1','c2','c3','c4','c5','c6','c7','c8'], order=True)

        print(ivList)

        return ivList

    def test_iv_select(self):
        """
        功能描述：测试iv_select函数

        管理记录：
        1. edited by 王文丹 2021/07/24
        """
        del self.datafake['certid']
        (ivDropVar,IvWoeData,ivList) = FeaDisSelectClass(self.ivThreshold,self.xgbSelectNum).iv_select(self.datafake,'label')
        print(ivDropVar)
        print(IvWoeData)
        print(ivList)
        return (ivDropVar,IvWoeData,ivList)    

    # train = pd.read_csv('../code/train.csv') # 自己导入一个测试数据
    # test= pd.read_csv('../code/test.csv')
    
    # ivThreshold = 0.02
    # xgbSelectNum = 13
    # target = 'TARGET'
    
    # def test_xgb_select(self):
    #     disselectclass = FeaDisSelectClass(self.ivThreshold,self.xgbSelectNum)
    #     disselectclass.xgb_select = mock.Mock(return_value =(1,2,3) , side_effect = disselectclass.xgb_select)

    #     (xgbdrop, woedata, xgbfea) = disselectclass.xgb_select(self.train,self.test,self.target)
    #     disselectclass.xgb_select.assert_called_with(self.train,self.test,self.target)  # 测试对应的输入变量是否能对应上       
    #     a1 =  self.assertTrue(len(xgbdrop)>0, msg=None)
    #     a2 = self.assertTrue(xgbfea.shape[0]>0, msg=None)
    #     print(xgbfea)

    #     return(a1,a2)

if __name__ == '__main__': 
    
    unittest.main() 



