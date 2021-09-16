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

    # 调起要测试的类   
    Transformclass = C_PreprocessData.TransformValueClass()

    def test_transform_object_dict(self):

        """
        功能描述： 测试函数transform_object_dict
        """
        # 数据构造
        listObject = ['特征2','特征3']
        ylabel = 'y'
        
        self.Transformclass.transform_object_dict = mock.Mock(return_value = 1,side_effect=self.Transformclass.transform_object_dict) #给输出函数定义输出结果
        # transform_object_dict的测试调用
        result = self.Transformclass.transform_object_dict(self.dataTest,listObject,ylabel)
        print(result)  # 直接输出查看
        
        self.Transformclass.transform_object_dict.assert_called_with(self.dataTest,listObject,ylabel)  # 测试对应的输入变量是否能对应上
        a1 = self.assertEqual(len(result),2)   # 验证输出的list长度是否对应的上

        return (result,a1)

    def test_transform_object_type(self):

        """
        功能描述：测试将数据集众的字符型变量转换为数值型变量 transform_object_type
        
        """

        # 数据构造
        listObject = ['特征2','特征3']
        dictObjectMapping = {'特征2': {'A': 3.0, 'C': 5.0, 'D': 1.0, 'F': 2.0, 'Y': 4.0}, '特征3': {'A': 4.0, 'B': 2.0, 'C': 3.0, 'D': 1.0}}

        self.Transformclass.transform_object_type = mock.Mock(side_effect=self.Transformclass.transform_object_type) #给输出函数定义输出结果

        # transform_object_dict的测试调用
        result = self.Transformclass.transform_object_type(self.dataTest,listObject,dictObjectMapping)
        print(result)  # 直接输出查看

        self.Transformclass.transform_object_type.assert_called_with(self.dataTest,listObject,dictObjectMapping)  # 测试对应的输入变量是否能对应上

        a1 = self.assertEqual(result.shape[0],26)   # 验证输出的list长度是否对应的上
        a2 = self.assertEqual(result.shape[1],7)   # 验证输出的list长度是否对应的上

        return (a1, a2)


if __name__ == '__main__':  
    unittest.main()





    
