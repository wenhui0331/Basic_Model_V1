import pandas as pd
import unittest
from unittest import TestCase
from unittest import mock
import C_PreprocessData

class TestDictValueHandle(TestCase):

    """
    功能描述：函数字典转换测试模块

    测试涉及的函数：dict_transform
    
    """

    def test_dict_transform(self):
        """

        功能描述：测试函数dict_transform

        """

        # 导入测试数据
        data = pd.read_csv('D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/sxk_person_model_sample20210407.csv')
        data1Dict = pd.read_excel('D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/指标翻译.xlsx')# 导入字典项目
        nameMappingLeft = '变量名'
        nameMappingRight = '解释'

        # 调起要测试的类
        # 测试对应的结果是否正常，给输出函数定义一个输出结果
        C_PreprocessData.DictTransformClass().dict_transform = mock.Mock(return_value = 1 , side_effect = C_PreprocessData.DictTransformClass().dict_transform)

        # 检查输入参数
        # dictclass.dict_transform.assert_called_with(data,data1Dict)

        # really go : 真实测试调用
        result = C_PreprocessData.DictTransformClass().dict_transform(data,data1Dict,nameMappingLeft,nameMappingRight)

        # 验证输出的DataFrame行数是否对应
        a1 = self.assertEqual(result.shape[0],38037)   

        # 验证输出的DataFrame列数是否对应
        a2 = self.assertEqual(result.shape[1],323) 

        # 直接输出查看
        print(result)

        return (a1,a2)

if __name__ == '__main__':  
    unittest.main()
