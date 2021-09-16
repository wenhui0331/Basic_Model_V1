import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import unittest
from unittest import TestCase
from H_ModelEvaluateData.ModelStabilityModule.ModelStabilityHandle import ModelStabilityClass


class testModelStabilityClass(TestCase):

    """
    功能描述：测试模型稳定性
    
    """
    
    # 导入测试数据
    data_path = "D:\\hean016839\\建模工程包\\Basic_Strategy_Model\\test_file\\data\\data.csv"
    data = pd.read_csv(data_path, low_memory=False) # 自己导入一个测试数据
    train, oot = train_test_split(data, test_size=0.3, random_state=99)
    flagname = '是否违约'
    factorname = 'scores'
    cut_method = 'quantile'
    nbin = 20
    cutpoint = []
    
    def test_get_psi(self):

        """
        功能描述： 测试函数 get_psi
        """

        # 调起要测试的类
        rr = ModelStabilityClass() 
        
        (r1,r2) = rr.get_psi(self.train,self.oot,self.flagname,self.factorname,self.cut_method,self.nbin,self.cutpoint) 
        self.assertEqual(r2.shape[1], 5)   
        
        return r1
   
if __name__ == '__main__':  
    unittest.main()