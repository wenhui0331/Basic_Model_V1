from unittest import result
import pandas as pd
import numpy as np 
import unittest
from unittest import TestCase
from unittest import mock
from scipy import stats
from unittest.mock import patch
from time import *
import warnings
warnings.filterwarnings('ignore')
from F_SelectChar.FeaSelectProcessModule.FeaLinearSelectHandle import FeaLinearSelectClass # 导入需要测试的单元 
from Z2_TestFunction.FakeTestDataModule.FakeTestDataHandle import FakeTestDataClass 

class TestFeaLinearSelectClass(TestCase):
    
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
    corrThreshold = 0.002 # 因为随机造数 相关性会来的小些
    vifThreshold = 1.001  # 因为随机造数 膨胀系数会来的小些
    target = 'label'
    del datafake['certid']
    
    def test_corr_select(self):

        """
        功能描述：测试函数test_vif_select
        edited by 王文丹 at 2021/07/25
        
        """
        
        self.datafake = self.datafake.astype('float')
        (corrDropVar,corrData,corrDesc) = FeaLinearSelectClass(self.corrThreshold,self.vifThreshold).corr_select(self.datafake,self.target,'iv')

        print(corrDropVar)
        print(corrData)
        print(corrDesc)

        return (corrDropVar,corrData,corrDesc)

    def test_vif_select(self):
        """
        功能描述：测试函数vif_select
        edited by 王文丹 at 2021/07/25
        
        """
        self.datafake = self.datafake.astype('float')
        (vifDropVar ,vifData,vifList) = FeaLinearSelectClass(self.corrThreshold,self.vifThreshold).vif_select(self.datafake,self.target)

        print(vifDropVar)
        print(vifData)
        print(vifList)

        return (vifDropVar ,vifData,vifList)


#python -m unittest J_unintest/F_SelectChar_test/test_FeaSelectProcessModule/test_FeaLinearSelectHandle.py


    # def test_vif_select(self):
    #     """
    #     功能描述：测试函数test_vif_select
    #      edited by 唐 
    #     """
    #     linearselectclass = FeaLinearSelectClass(self.corrThreshold,self.vifThreshold)
    #     linearselectclass.vif_select = mock.Mock(return_value =(1,2,3) , side_effect = linearselectclass.vif_select) #给输出函数定义输出结果

    #     # output_var_type的测试调用
    #     (vifdrop, vifdata, vifdf) = linearselectclass.vif_select(self.train,self.target)
    #     linearselectclass.vif_select.assert_called_with(self.train,self.target)  # 测试对应的输入变量是否能对应上
    #     a1 =  self.assertEqual(vifdf.shape[1] ,2)
        
    #     print(vifdf.shape)  # 直接输出查看
    #     print(vifdrop)
    #     print(vifdata)

    #     return (a1)    


    # @patch("F_SelectChar.FeaDisSelectClass.iv_claculate")
   
    # def test_corr_select(self,mock_iv_claculate):
        #   """
        #  edited by 唐
          
        #   """    
    #     linearselectclass = FeaLinearSelectClass(self.corrThreshold,self.vifThreshold)
    #     #linearselectclass.corr_select = mock.Mock(return_value =(1,2,3) , side_effect = linearselectclass.corr_select) #给输出函数定义输出结果

    #     ivlist = pd.read_csv('../code/ivList.csv')
    #     mock_iv_claculate.return_value = (ivlist)
    #     # output_var_type的测试调用
    #     (list1, woedata,corrdec) = linearselectclass.corr_select(self.train,self.target)
    #     a1 = self.assertEqual(woedata.shape[0],6896)   # 验证输出的DataFrame行数是否对应
    #     a2 = self.assertEqual(woedata.shape[1],23) # 验证输出的DataFrame列数是否对应
    #     print(woedata.shape)  # 直接输出查看

    #     return (a1,a2)

if __name__ == '__main__': 
    
    unittest.main() 





