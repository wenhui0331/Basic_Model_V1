import C_PreprocessData
from unittest import result
import pandas as pd
import numpy as np 
import unittest
from unittest import TestCase
from unittest import mock
from scipy import stats
from unittest.mock import patch



class TestMissValueHandle(TestCase):

    """
    功能描述：缺失值处理测试类
    """

    dataTest = pd.read_excel('D:/data/基础测试数据/1.xlsx') # 自己导入一个测试数据
    thresholdMissingDele = 0.8
    styleFillingVar = '1'


    def test_output_var_type(self): # 本测试函数只是提供了一个测试的场所

       
        
        missclass =  C_PreprocessData.MissValueClass(self.thresholdMissingDele,self.styleFillingVar) # 调起要测试的类
        missclass.output_var_type = mock.Mock(return_value =(1,2,3) , side_effect = missclass.output_var_type) #给输出函数定义输出结果

        # output_var_type的测试调用
        (result, list1, list2) = missclass.output_var_type(self.dataTest)

        # missclass.output_var_type.assert_called_with(self.dataTest)  # 测试对应的输入变量是否能对应上
        a1 = self.assertEqual(result.shape[0],7)   # 验证输出的DataFrame行数是否对应
        a2 = self.assertEqual(result.shape[1],10) # 验证输出的DataFrame列数是否对应
        print(result)  # 直接输出查看

        return (a1,a2)
    

    @patch("C_PreprocessData.MissValueModule.MissValueClass.output_var_type")
    def test_missing_cal(self,mock_output_var_type):

        missclass = C_PreprocessData.MissValueClass(self.thresholdMissingDele,self.styleFillingVar) # 调起要测试的类

        # 2. 给中间被调用的函数定义一个数据  (核心函数)
        data_middle = pd.read_excel('D:/data/基础测试数据/1_mock测试结果.xlsx')
        listObject = ['yyyyy','yyyyyy']
        listNum = [1,2]
        mock_output_var_type.return_value = (data_middle,listObject,listNum)

        # 3. missing_cal的测试调用
        result = missclass.missing_cal(self.dataTest)
        # missclass.missing_cal.assert_called_with(self.dataTest)  # 测试对应的输入变量是否能对应上
        a1 = self.assertEqual(result.shape[0],7)   # 验证输出的DataFrame行数是否对应 也可以进行其他验证
        a2 = self.assertEqual(result.shape[1],13)  # 验证输出的DataFrame列数是否对应
        print(result)  # 直接输出查看
        return (a1,a2)

    
    @patch("C_PreprocessData.MissValueModule.MissValueClass.missing_cal")
    def test_missing_delete_var(self,mock_missing_cal):

        # 1. 导入测试数据
        # dataTest = pd.read_excel('D:/data/基础测试数据/1.xlsx') # 自己导入一个测试数据
        missclass = C_PreprocessData.MissValueClass(self.thresholdMissingDele,self.styleFillingVar) # 调起要测试的类

        # 2. 给中间被调用的函数定义一个数据  (核心函数)
        data_middle = pd.read_excel('D:/data/基础测试数据/2_mock测试结果.xlsx')
        mock_missing_cal.return_value = data_middle

        # 3. missing_cal的测试调用
        (result1,result2) = missclass.missing_delete_var(self.dataTest)
        # missclass.missing_cal.assert_called_with(dataTest)  # 测试对应的输入变量是否能对应上
        a1 = self.assertEqual(len(result1),1)   # 验证输出的DataFrame行数是否对应 也可以进行其他验证
        a2 = self.assertEqual(len(result2),6)  # 验证输出的DataFrame列数是否对应
        print(result1)  # 直接输出查看
        print(result2)

    def test_missing_fill_var(self):

        missclass =C_PreprocessData.MissValueClass(self.thresholdMissingDele,self.styleFillingVar) # 调起要测试的类
        missclass.missing_fill_var = mock.Mock(return_value =(1,2,3) , side_effect = missclass.missing_fill_var) #给输出函数定义输出结果

        # output_var_type的测试调用
        result= missclass.missing_fill_var(self.dataTest)
        missclass.missing_fill_var.assert_called_with(self.dataTest)  # 测试对应的输入变量是否能对应上
        a1 = self.assertEqual(result.shape[0],26)   # 验证输出的DataFrame行数是否对应
        a2 = self.assertEqual(result.shape[1],7) # 验证输出的DataFrame列数是否对应
        print(result)  # 直接输出查看

        return (a1,a2)

if __name__ == '__main__': 

    unittest.main()










