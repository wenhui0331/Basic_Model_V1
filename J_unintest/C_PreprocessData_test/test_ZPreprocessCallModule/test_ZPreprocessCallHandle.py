import pandas as pd
import unittest
from unittest import TestCase
from unittest import mock
import os
import C_PreprocessData
import json
from A_GetData.DocumentsInputModule.DocumentsInputHandle import DocumentsInputClass

class TestZPreprocessCallClass(TestCase):
    
    def test_Preprocess_all_Call(self):

        # 导入json配置文件
        path = os.getcwd().replace('\\','/')+'/C_PreprocessData/ZPreprocessCallModule/ParaConfig.json'
        f = open(path)
        config_parameter = json.load(f)
        print("加载配置参数完成...")
        print(config_parameter)

        # 1. 数据参数
        thresholdMissingDele = config_parameter['thresholdMissingDele'] #缺失删除阈值 
        styleFillingVar = config_parameter['styleFillingVar'] #缺失填充方式 int (1:中位数,2:众数,3:均值,4:统一填充missing)
        thresholdSingleDrop = config_parameter['thresholdSingleDrop']  #单一值处理阀值 float
        listNeedTransform = config_parameter['listNeedTransform']  #需要数据类型转换的字符型变量 list
        ylabel = config_parameter['ylabel']   #坏样本率对应的特征名称 （如：'y')
        listDropRepeat = config_parameter['listDropRepeat'] # 确认数据集是否重复的字段列表 list
        objectNeedRemove = config_parameter['objectNeedRemove']
        NumNeedRemove = config_parameter['NumNeedRemove']
        pathSaved = config_parameter['pathSaved'] # 数据清洗结果的存储路径 (如：'D:/1.csv' str)

        # 调起数据清洗的类
        callclass = C_PreprocessData.ZPreprocessCallClass(thresholdMissingDele,styleFillingVar,thresholdSingleDrop,listNeedTransform,ylabel,listDropRepeat,objectNeedRemove,NumNeedRemove,pathSaved)

        dataPath = 'D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/sxk_person_model_sample20210407.csv'
        dataStyle = 'csv'
        dataDict = pd.read_excel('D:/1. 泰隆工作/1. 工作资料/3.税享卡/税享卡-微众税银-建模/数据样本/指标翻译.xlsx')# 导入字典项目
        nameMappingLeft = '变量名'
        nameMappingRight = '解释'

        dataForPreprocess1 = callclass.Preprocess_all_Call(dataPath,dataStyle,dataDict,nameMappingLeft,nameMappingRight)

        return dataForPreprocess1



if __name__ == '__main__':  
    unittest.main()








