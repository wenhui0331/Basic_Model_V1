import pandas as pd
import numpy as np

class TransformValueClass():

    """
    功能描述：字符型变量向数值型变量转换

    输入：
    dataForPreprocess：预处理前的数据集 DataFrame 
    listObject: 需要数据类型转换的字符型变量 list
    ylabel: 坏样本率对应的特征名称 （如：'y')

    输出：
    函数transform_object_dict: 主要用于字符型变量转换为数值型变量，依照坏样本率排序，形成映射字典
    函数transform_object_type：将数据集众的字符型变量转换为数值型变量


    管理记录：
    1. edited by 王文丹 2021/07/02
    
    """

    def output_var_type(self,dataForPreprocess):
        """

        功能描述：主要功能识别字符型变量
        
        """
        dataDtypes = dataForPreprocess.dtypes.reset_index(level = 0)
        dataDtypes.rename(columns = {'index':'变量名称',0:'数据类型'}, inplace = True)

        data_describe = dataForPreprocess.describe().T.reset_index(level = 0)
        data_describe.rename(columns={'index':'变量名称'}, inplace = True)

        dataTypes = pd.merge(dataDtypes,data_describe,on='变量名称',how='left')

        listObject = list(dataTypes[dataTypes['count'].isnull()]['变量名称'])
        listNum = list(dataTypes[dataTypes['count'].notnull()]['变量名称'])
        
        return (dataTypes,listObject,listNum)

    
    def transform_object_dict(self,dataForPreprocess,listNeedTransform,ylabel):

        """
        功能描述：主要用于字符型变量转换为数值型变量，依照坏样本率排序，形成映射字典

        输入：
        dataForPreprocess：预处理前的数据集 DataFrame
        listNeedTransform: 需要数据类型转换的字符型变量 list
        ylabel: 坏样本率对应的特征名称 （如：'y')

        输出：
        dictObjectMapping：字符型变量映射为数值型变量的字典 dict

        管理记录：
        1. edited by 王文丹 2021/07/02
        
        """

        try:

            dictObjectMapping = {}

            for colObject in listNeedTransform:
                
                dr = dataForPreprocess[[colObject,ylabel]]
                dr1 = dr.groupby([colObject]).agg({ylabel:['count','sum']})
                dr1['badrate']=dr1[ylabel,'sum']/dr1[ylabel,'count']
                dr1['sort_num'] = dr1['badrate'].rank(ascending = 1,method = 'first')
                dic1 = {}
                mapping= {}
                for key,value in zip(dr1.index,dr1['sort_num']):
                    mapping[key] = value
                    
                dictObjectMapping[colObject] = mapping
            print('{0}{1}' . format ( '字符串变量转换为数值型的映射关系（按坏样本率排序）: ',dictObjectMapping))

            return dictObjectMapping

        except Exception as e:

            print('C_PreprocessData,TransformValueModule,TransformValueHandle,TransformValueClass,transform_object_dict',':',e)

        
    def transform_object_type(self,dataForPreprocess,listNeedTransform,dictObjectMapping):

        """
        功能描述：将数据集众的字符型变量转换为数值型变量

        输入：
        dataForPreprocess：预处理前的数据集 DataFrame
        dictObjectMapping: 字符型变量映射为数值型变量的字典 dict
        listNeedTransform: 需要数据类型转换的字符型变量 list

        输出：
        dataForPreprocess：字符转换后的数据集 DataFrame

        管理记录：
        1. edited by 王文丹 2021/07/02

        """
        try:
            for i in listNeedTransform:
                dataForPreprocess[i] = dataForPreprocess[i].map(dictObjectMapping[i])
            return dataForPreprocess
        
        except Exception as e:

            print('C_PreprocessData,TransformValueModule,TransformValueHandle,TransformValueClass,transform_object_type',':',e)

