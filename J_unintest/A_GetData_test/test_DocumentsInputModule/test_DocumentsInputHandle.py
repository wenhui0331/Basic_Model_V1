import pandas as pd
import unittest
from unittest import TestCase
from unittest import mock
import A_GetData  # 导入需要测试的函数包

class TestDocumentsInputHandle(TestCase):

    """
    功能描述：文档导入式的数据读取方法

    测试涉及的函数：document_input
    
    """

    def test_document_input(self):

        """
        功能描述：测试函数document_input
        
        """
        # 导入数据路径
        data = 'D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/sxk_person_model_sample20210407.csv'

        # 导入数据格式
        dataStyle = 'csv'

         # 测试对应的结果是否正常，给输出函数定义一个输出结果  side_effect 可以覆盖return_value的值
        A_GetData.DocumentsInputClass().document_input = mock.Mock(return_value = 1 , side_effect = A_GetData.DocumentsInputClass().document_input)

        # really go : 真实测试调用
        result = A_GetData.DocumentsInputClass().document_input(data,dataStyle)

        # 验证输出的DataFrame行数是否对应
        a1 = self.assertEqual(result.shape[0],38037) 

        # 验证输出的DataFrame列数是否对应
        a2 = self.assertEqual(result.shape[1],323)  # 也可以直接进行其他的数值对比，这边只是为了测试函数是否正常运行

        # 直接输出查看 更直观
        print(result)

        return (a1, a2)

if __name__ == '__main__':  
    unittest.main()





