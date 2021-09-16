import pandas as pd
import numpy as np
from scipy import stats


class MissValueClass():

    """  
    功能：缺失值处理

    输入：
    dataForPreprocess: 数据集 pandas 
    thresholdMissingDele: 缺失删除阈值 float
    styleFillingVar: 缺失填充方式 str ('1':统一填充为missing,'2':中位数,'3':均值,'4':众数)

    输出：
    函数output_var_type: 变量概况，包括数据类型和基础统计描述，如标准差和分位数等
    函数missing_cal: 变量概况，包括数据类型和基础统计描述，如标准差、分位数、数据缺失情况等
    函数missing_delete_var：缺失值处理函数，缺失率高于阀值，将该变量删除
    函数missing_fill_var：缺失值填充函数


    维护记录：
    1. edited by 王文丹 2021/07/02   

    """

    def __init__(self,thresholdMissingDele,styleFillingVar,specialValue):

        self.thresholdMissingDele = thresholdMissingDele   #缺失删除阈值 float
        self.styleFillingVar = styleFillingVar             #缺失填充方式 str
        self.specialValue = specialValue                   #特殊值处理


    def output_var_type(self,dataForPreprocess):
        """
        功能描述：变量概况，包括数据类型和基础统计描述，如标准差和分位数等

        输入：
        dataForPreprocess: 预处理前的数据集 DataFrame 
        
        输出:
        dataDtypes: 变量描述 DataFrame 
        listObject: 字符型变量列表 list
        listNum: 数值型变量列表 list

        管理记录：

        1. edited by 王文丹 2021/07/02
        
        """
        try:
            
            dataDtypes = dataForPreprocess.dtypes.reset_index(level = 0)
            dataDtypes.rename(columns = {'index':'变量名称',0:'数据类型'}, inplace = True) 
            
            data_describe = dataForPreprocess.describe().T.reset_index(level = 0)
            data_describe.rename(columns={'index':'变量名称'}, inplace = True)
            
            dataDtypes = pd.merge(dataDtypes,data_describe,on='变量名称',how='left')
            listObject = list(dataDtypes[dataDtypes['count'].isnull()]['变量名称'])
            listNum = list(dataDtypes[dataDtypes['count'].notnull()]['变量名称'])

            return (dataDtypes,listObject,listNum)

        except Exception as e:

            print('C_PreprocessData, MissValueModule, MissValueHandle, MissValueClass, output_var_type',':',e)
    

    def missing_cal(self,dataForPreprocess):

        """
        功能描述：变量概况，包括数据类型和基础统计描述，如标准差、分位数、数据缺失情况等

        输入：
        dataForPreprocess: 预处理前的数据集 DataFrame

        输出:
        dataTypesMissing: 变量描述(加上数据缺失情况) DataFrame 

        管理记录：
        1. edited by 王文丹 2021/07/02
        
        """

        try:
        
            dataTypes = self.output_var_type(dataForPreprocess)[0]
            dataMissing = dataForPreprocess.isnull().sum().reset_index(level = 0)
            dataMissing.rename(columns={'index':'变量名称',0: '缺失值个数'}, inplace = True)
            dataMissing['缺失率'] = dataMissing['缺失值个数']/dataForPreprocess.shape[0]
            dataTypesMissing = pd.merge(dataTypes,dataMissing,on='变量名称',how='left')       
            return dataTypesMissing

        except Exception as e:

            print('C_PreprocessData, MissValueModule, MissValueHandle, MissValueClass, missing_cal',':',e)


    def missing_delete_var(self,dataForPreprocess):

        """
        缺失值删除函数：缺失率高于阀值，将该变量删除

        输入：
        dataForPreprocess: 预处理前的数据集 DataFrame
        thresholdMissingDele：缺失删除阈值 float

        输出：
        colMissingOverThreshold：缺失率太高被删除的特征 list
        remainColumns：遗留特征列表 list

        管理记录：
        1. edited by 王文丹 2021/07/02
      
        """
        try:
            remainColumns = list(dataForPreprocess.columns)  # 原有的所有变量名称     
            dataTypesMissing = self.missing_cal(dataForPreprocess)    
            colMissingOverThreshold = list(dataTypesMissing[dataTypesMissing['缺失率']>=self.thresholdMissingDele]['变量名称'])
            
            print( '{0}{1}{2}{3}' . format ( '缺失值高于阀值',self.thresholdMissingDele, '的变量个数：', len(colMissingOverThreshold)))
            print( '{0}{1}' . format ( '缺失值高于阀值变量名称：',colMissingOverThreshold))
            
            for col in colMissingOverThreshold:
                remainColumns.remove(col)
                        
            print( '{0}{1}{2}{3}' . format ( '缺失值低于阀值',self.thresholdMissingDele, '的变量个数：', len(remainColumns)))
            print( '{0}{1}' . format ( '缺失值低于阀值变量名称：',remainColumns))
            
            return (colMissingOverThreshold,remainColumns)

        except Exception as e:

            print('C_PreprocessData, MissValueModule, MissValueHandle, MissValueClass, missing_delete_var',':',e)


    def missing_fill_var(self,dataForPreprocess,listneedFilling):
    
        """
        功能描述：缺失值填充函数
        
        输入：
        dataForPreprocess: 预处理前的数据集 DataFrame
        listneedFilling: 需要被填充的字段 list
        styleFillingVar: 缺失填充方式 str ('1':统一填充为missing,'2':中位数,'3':均值,'4':众数)

        输出：
        dataMissfilled：缺失值填充完成的数据集

        管理记录：
        1. edited by 王文丹 2021/07/02

        """

        try:
        
            #  统一填充为'missing'
            if self.styleFillingVar == '1':
                dataForPreprocess[listneedFilling] = dataForPreprocess[listneedFilling].fillna(self.specialValue).replace('\\N',self.specialValue)
        
            #  中位数填充
            elif self.styleFillingVar == '2':
                dataForPreprocess[listneedFilling] = dataForPreprocess[listneedFilling].fillna(dataForPreprocess[listneedFilling].median())
                
            # 均值填充
            elif self.styleFillingVar == '3':
                dataForPreprocess[listneedFilling] = dataForPreprocess[listneedFilling].fillna(dataForPreprocess[listneedFilling].mean())
            
            # 众数填充
            elif self.styleFillingVar == '4':
                mostValue = pd.DataFrame(stats.mode(dataForPreprocess[listneedFilling])[0][0])
                mostValue.index =  dataForPreprocess[listneedFilling].columns
                dataForPreprocess[listneedFilling] = dataForPreprocess[listneedFilling].fillna(mostValue[0])
        
            return dataForPreprocess 

        except Exception as e:

            print('C_PreprocessData, MissValueModule, MissValueHandle, MissValueClass, missing_fill_var',':',e)




    




