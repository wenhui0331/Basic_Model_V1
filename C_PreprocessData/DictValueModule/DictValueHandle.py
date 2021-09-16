import pandas as pd
class DictTransformClass():

    """
    功能描述：函数字典转换

    输入：
    dataForPreprocess：预处理前的数据集 DataFrame 
    dataDict: 字典项目 DataFrame

    输出：
    dataForPreprocess: 字典转换后的数据集 DataFrame

    管理记录：
    1. edited by 王文丹 2021/07/02

    """
    def dict_transform(self,dataForPreprocess,dataDict,nameMappingLeft,nameMappingRight):

        """
        功能描述：字典转换

        输入：
        dataForPreprocess：预处理前的数据集 DataFrame 
        dataDict: 字典数据 DataFrame
        nameMappingLeft: 字典中的变量名称（一般为英文,'变量名'）
        nameMappingRight: 字典中的变量名称解释（一般为中文释义,'解释'）


        输出：
        dataForPreprocess: 字典转换后的数据集 DataFrame

        管理记录：
        1. edited by 王文丹 2021/07/02
        
        """
    
        try:

            mapping2 = {}
            
            for key,value in zip(dataDict[nameMappingLeft],dataDict[nameMappingRight]):
                mapping2[key] = value
            dataForPreprocess.rename(columns=mapping2,inplace=True)

            return dataForPreprocess

        except Exception as e:

            print('C_PreprocessData,DictValueModule,DictValueHandle,DictTransformClass,dict_transform',':',e)