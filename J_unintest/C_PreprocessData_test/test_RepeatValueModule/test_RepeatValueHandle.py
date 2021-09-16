import pandas as pd
import numpy as np 
import unittest
from unittest import result
from unittest import TestCase
from unittest import mock
import C_PreprocessData


class TestTransformValueClass(TestCase):

    """
    功能描述：字符型转换为数值型变量测试函数
    
    """
    # 导入测试数据

    dataTest = pd.read_excel('D:/data/基础测试数据/1.xlsx') # 自己导入一个测试数据
    thresholdSingleDrop = 0.9

    # 调起要测试的类
    
    Repeatclass = C_PreprocessData.RepeatValueModule.RepeatValueClass(thresholdSingleDrop)

    def test_repeat_drop_value(self):

        """
        功能描述： 测试函数repeat_drop_value
        """
        
        listDropRepeat = ['特征1','特征2'] # 数据构造
        self.Repeatclass.repeat_drop_value = mock.Mock(return_value = 1,side_effect=self.Repeatclass.repeat_drop_value) #给输出函数定义输出结果
        result = self.Repeatclass.repeat_drop_value(self.dataTest,listDropRepeat) # transform_object_dict的测试调用
        self.Repeatclass.repeat_drop_value.assert_called_with(self.dataTest,listDropRepeat)  # 测试对应的输入变量是否能对应上
        a1 = self.assertEqual(result.shape[0],20)   # 验证输出的list长度是否对应的上
        print(result)  # 直接输出查看

        return (result,a1)

    def test_single_drop_value(self):

        """
        功能描述： 测试函数single_drop_value 单一值处理
        """
        
        listDropSingle = self.dataTest.columns # 数据构造

        self.Repeatclass.single_drop_value = mock.Mock(return_value = 1,side_effect=self.Repeatclass.single_drop_value) #给输出函数定义输出结果
        (result1,result2) = self.Repeatclass.single_drop_value(self.dataTest,listDropSingle) # transform_object_dict的测试调用
        self.Repeatclass.single_drop_value.assert_called_with(self.dataTest,listDropSingle)  # 测试对应的输入变量是否能对应上
        a1 = self.assertEqual(result2.shape[0],26)   # 验证输出的list长度是否对应的上
        print(result1)  # 直接输出查看
        print(result2)  # 直接输出查看

        return (result,a1)
   
if __name__ == '__main__':  
    unittest.main()
