import os 
import sys 
sys.path.append(os.getcwd())

import G_ModelEstablish
from G_ModelEstablish.ScorecardEstablishModule.ScorecardEstablishHandle import ScorecardEstablishClass
import pandas as pd
import numpy as np
import unittest
from unittest import TestCase
from unittest import mock
import warnings 
warnings.filterwarnings("ignore")
from time import *

class TestScorecardEstablishClass(TestCase):
    def test_scorecard_establishment(self):

        # 设置测试数据
        
        dataPath = '/Users/mwh/Desktop/泰隆/工作内容/结构化/资料/样本/文丹/sxk_person_model_sample20210407_alter.csv'
        dataStyle = 'csv'
        dataDictPath = '/Users/mwh/Desktop/泰隆/工作内容/结构化/资料/样本/文丹/指标翻译.xlsx'
        dataPathStyle = 'excel'
        nameMappingLeft = '变量名'
        nameMappingRight = '解释'

        thresholdMissingDele = 0.9  #缺失删除阈值 
        styleFillingVar = '1' #缺失填充方式 int ('1':统一填充为missing,'2':中位数,'3':均值,'4':众数)
        specialValue = -9999999
        thresholdSingleDrop = 0.9  #单一值处理阀值 float
        listNeedTransform = ['sample_type'] #需要数据类型转换的字符型变量 list
        ylabel = 'label'   #坏样本率对应的特征名称 （如：'y')
        listDropRepeat = ['busi_ctr_id','dt']# 确认数据集是否重复的字段列表 list
        objectNeedRemove = ['busi_ctr_id','dt','name_sc','code_sc','samp_type','report_no'] # 需要被删除的字符型变量
        NumNeedRemove = ['label','apply_month'] # 需要被删除的数值型变量
        pathSaved = '/Users/mwh/Desktop/泰隆/工作内容/结构化/资料/样本/文丹/' # 数据清洗结果的存储路径 (如：'D:/1.csv' str)
        maxNumBox = 5
        dataSeparation = {'区分类型字段':'samp_type','训练集合':'dev','测试集合':'val','样本外集合':'oot'}

        # dataPath = 'D:/15. 建模包测试结果/QYCredit_data1.xlsx'
        # dataPath = 'D:/15. 建模包测试结果/企业征信排查错误.xlsx'
        # dataStyle = 'excel'
        # dataDictPath = None   # 不需要字典转换
        # dataPathStyle = None  # 不需要字典转换
        # nameMappingLeft = None  # 不需要字典转换
        # nameMappingRight = None # 不需要字典转换


        # thresholdMissingDele = 0.9  #缺失删除阈值 
        # styleFillingVar = '1' #缺失填充方式 int ('1':统一填充为missing,'2':中位数,'3':均值,'4':众数)
        # specialValue = -999999
        # thresholdSingleDrop = 0.9  #单一值处理阀值 float
        # listNeedTransform = ['QY_查询原因'] #需要数据类型转换的字符型变量 list
        # listNeedTransform = [] #需要数据类型转换的字符型变量 list
        # ylabel = 'y'   #坏样本率对应的特征名称 （如：'y')
        # listDropRepeat = ['bus_ctr_id']# 确认数据集是否重复的字段列表 list
        # objectNeedRemove = ['bus_ctr_id', 'QY_证件类型', 'QY_证件号码', 'QY_报告日期', 'QY_报告编号','samp_type'] # 需要被删除的字符型变量
        # NumNeedRemove = ['QY_客户号','dt', '企业征信_gap_days', '企业征信_rank', 'y','月份'] # 需要被删除的数值型变量
        # pathSaved = '/Users/mwh/Desktop/泰隆/工作内容/结构化/资料/样本/文丹/' # 数据清洗结果的存储路径 (如：'D:/1.csv' str)
        # maxNumBox = 5
        # dataSeparation = {'区分类型字段':'samp_type','训练集合':'train','测试集合':'test','样本外集合':'oot'}

        binMethod = 'chiCutMethod'
        cutParameters = {'最小单箱占比':0.05,'卡方值置信度':0.95, '卡方值自由度':4, '是否单调':1, '初始分箱数':50}
        needModifyDict = {}
        ivThreshold = 0.02
        xgbSelectNum=None
        psiThreshold=0.01
        corrThreshold=0.6
        pThreshold=0.05
        selection='stepwise'
        sle=0.05
        sls=0.05
        includes=[]
        vifThreshold=10

        baseScore = 500 #基础分数
        pdo = 20 #分数刻度

        numForSplit = 20 #score值展示分Bin
        preThreshold = 0.05 #经验定义的坏样本的阈值 


        # 重新定义类
        SEC = ScorecardEstablishClass(thresholdMissingDele,styleFillingVar, specialValue,thresholdSingleDrop,listNeedTransform,ylabel,listDropRepeat,objectNeedRemove,NumNeedRemove,pathSaved,maxNumBox,binMethod,cutParameters,needModifyDict,ivThreshold,xgbSelectNum,psiThreshold,corrThreshold,vifThreshold,pThreshold,selection,sle,sls,includes,baseScore,pdo,numForSplit,preThreshold)
        (trainForLr,testForLr) = SEC.scorecard_establishment(dataPath,dataStyle,dataDictPath,dataPathStyle,nameMappingLeft,nameMappingRight,dataSeparation)
        return trainForLr,testForLr

if __name__ == '__main__':  
    unittest.main()


#执行路径：python -m unittest J_unintest/G_ModelEstablish_test/test_ScorecardEstablishModule/test_ScorecardEstablishHandle.py 