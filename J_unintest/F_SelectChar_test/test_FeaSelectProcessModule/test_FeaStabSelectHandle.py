from unittest import result
import pandas as pd
import numpy as np 
import unittest
from unittest import TestCase
from unittest import mock
from scipy import stats
from unittest.mock import patch
from time import *
# 导入需要测试的单元
from F_SelectChar.FeaSelectProcessModule.FeaStabSelectHandle import FeaStabSelectClass 
from Z2_TestFunction.FakeTestDataModule.FakeTestDataHandle import FakeTestDataClass


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
    for i in range(40):
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

    def test_psi_claculate(self):
        """
        功能描述：测试psi_claculate

        edited by 王文丹 2021/07/25
        
        """
        self.train = self.train.astype('float')
        self.test = self.test.astype('float')
        psiList = FeaStabSelectClass(self.psiThreshold).psi_claculate(self.train, self.test, self.target)
        print(psiList)
        return psiList

    def test_psi_select(self):
        """
        功能描述：测试pis_select

        edited by 王文丹 2021/07/25
        
        """
        self.train = self.train.astype('float')
        self.test = self.test.astype('float')
        psiThreshold = 0.0002  # 因为是随机造数 psi会小些
        (psiDropVar,psiData,psiList) = FeaStabSelectClass(psiThreshold).psi_select(self.train,self.test,self.target)
        print(psiDropVar)
        print(psiData)
        print(psiList)
        return (psiDropVar,psiData,psiList)

#python -m unittest J_unintest/F_SelectChar_test/test_FeaSelectProcessModule/test_FeaStabSelectHandle.py



    # def test_psi_select(self):
    #     staselectclass = FeaStabSelectClass(self.psiThreshold)
    #     staselectclass.psi_select = mock.Mock(return_value =(1,2,3) , side_effect = staselectclass.psi_select) #给输出函数定义输出结果

    #     # output_var_type的测试调用
    #     (psidrop, woedata, psidf) = staselectclass.psi_select(self.train,self.test,self.target)
    #     staselectclass.psi_select.assert_called_with(self.train,self.test,self.target)  # 测试对应的输入变量是否能对应上
    #     a1 =  self.assertTrue(woedata.shape[1] > 0, msg=None)
    #     a2 = self.assertTrue(psidf.shape[0]>0, msg=None)
    #     print(psidf.shape)  # 直接输出查看

    #     return (a1,a2)
    


if __name__ == '__main__': 
    
    unittest.main() 



