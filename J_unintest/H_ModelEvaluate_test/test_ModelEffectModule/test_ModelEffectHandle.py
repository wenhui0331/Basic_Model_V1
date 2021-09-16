import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import unittest
from unittest import TestCase
from H_ModelEvaluateData.ModelEffectModule.ModelEffectHandle import ModelEffectClass
import warnings
warnings.filterwarnings("ignore")


class testModelEffectClass(TestCase):

    """
    功能描述：测试模型稳定性
    
    """
    
    # 导入测试数据
    data_path = "D:\\hean016839\\建模工程包\\Basic_Strategy_Model\\test_file\\data\\data.csv"
    data = pd.read_csv(data_path, low_memory=False) # 自己导入一个测试数据
    train, oot = train_test_split(data, test_size=0.3, random_state=99)
    dev, val = train_test_split(train, test_size=0.3, random_state=99)
    flagname = '是否违约'
    factorname = 'scores'
    cut_method = 'quantile'
    nbin = 20
    cutpoint = []
    
    def test_get_ks(self):

        """
        功能描述： 测试函数 get_ks
        """

        # 调起要测试的类
        rr = ModelEffectClass() 
        
        r1 = rr.get_ks(self.dev,self.flagname,self.factorname) 
        self.assertTrue(r1<=1) 
        
        return r1
    
    def test_get_auc(self):

        """
        功能描述： 测试函数 get_auc
        """

        # 调起要测试的类
        rr = ModelEffectClass() 
        
        r1 = rr.get_auc(self.dev,self.flagname,self.factorname) 
        self.assertTrue(r1<=1) 
        
        return r1

    def test_get_gini(self):

        """
        功能描述： 测试函数 get_gini
        """

        # 调起要测试的类
        rr = ModelEffectClass() 
        
        r1 = rr.get_gini(self.dev,self.flagname,self.factorname) 
        self.assertTrue(r1<=1) 
        
        return r1
   
    def test_get_report(self):

        """
        功能描述： 测试函数 get_report
        """

        # 调起要测试的类
        rr = ModelEffectClass() 
        
        r1 = rr.get_report(self.dev,self.flagname,self.factorname,self.cut_method,self.nbin,self.cutpoint) 
        self.assertEqual(r1.shape[1], 17) 
        
        return r1

    def test_plot_ks(self):

        """
        功能描述： 测试函数 plot_ks
        """

        DfName = "训练集"
        ImagePath = "D:\\hean016839\\建模工程包\\Basic_Strategy_Model\\test_file\\result"

        # 调起要测试的类
        rr = ModelEffectClass() 
        
        rr.plot_ks(self.dev,self.flagname,self.factorname,DfName,ImagePath) 
        
    def test_plot_lift(self):

        """
        功能描述： 测试函数 plot_lift
        """

        ImagePath = "D:\\hean016839\\建模工程包\\Basic_Strategy_Model\\test_file\\result"

        # 调起要测试的类
        Repeatclass = ModelEffectClass() 
        
        Repeatclass.plot_lift(self.dev,self.val,self.oot,self.flagname,self.factorname,ImagePath) 

    def test_plot_pr(self):

        """
        功能描述： 测试函数 plot_pr
        """

        ImagePath = "D:\\hean016839\\建模工程包\\Basic_Strategy_Model\\test_file\\result"

        # 调起要测试的类
        Repeatclass = ModelEffectClass() 
        
        Repeatclass.plot_pr(self.dev,self.val,self.oot,self.flagname,self.factorname,ImagePath) 

    def test_plot_roc(self):

        """
        功能描述： 测试函数 plot_roc
        """

        ImagePath = "D:\\hean016839\\建模工程包\\Basic_Strategy_Model\\test_file\\result"

        # 调起要测试的类
        Repeatclass = ModelEffectClass() 
        
        Repeatclass.plot_roc(self.dev,self.val,self.oot,self.flagname,self.factorname,ImagePath) 

    def test_plot_pr_f1(self):

        """
        功能描述： 测试函数 plot_pr_f1
        """
        
        DfName = "训练集"
        ImagePath = "D:\\hean016839\\建模工程包\\Basic_Strategy_Model\\test_file\\result"

        # 调起要测试的类
        Repeatclass = ModelEffectClass() 
        
        Repeatclass.plot_pr_f1(self.dev,self.flagname,self.factorname,DfName,ImagePath) 

    def test_plot_ks_distribution(self):

        """
        功能描述： 测试函数 plot_ks_distribution
        """
        
        DfName = "训练集"
        ImagePath = "D:\\hean016839\\建模工程包\\Basic_Strategy_Model\\test_file\\result"

        # 调起要测试的类
        Repeatclass = ModelEffectClass() 
        
        Repeatclass.plot_ks_distribution(self.dev,self.flagname,self.factorname,DfName,ImagePath) 

    def test_plot_good_bad_ind(self):

        """
        功能描述： 测试函数 plot_good_bad_ind
        """
        
        DfName = "训练集"
        ImagePath = "D:\\hean016839\\建模工程包\\Basic_Strategy_Model\\test_file\\result"

        # 调起要测试的类
        Repeatclass = ModelEffectClass() 
        
        Repeatclass.plot_good_bad_ind(self.dev,self.flagname,self.factorname,DfName,ImagePath) 

    def test_plot_cumulative_approve(self):

        """
        功能描述： 测试函数 plot_cumulative_approve
        """
        
        DfName = "验证集"
        ImagePath = "D:\\hean016839\\建模工程包\\Basic_Strategy_Model\\test_file\\result"

        # 调起要测试的类
        Repeatclass = ModelEffectClass() 
        
        Repeatclass.plot_cumulative_approve(self.val,self.flagname,self.factorname,DfName,ImagePath) 

   
if __name__ == '__main__':  
    unittest.main()