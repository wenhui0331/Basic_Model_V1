import pandas as pd
import numpy as np

class RepeatValueClass():

    """
    功能描述：重复值处理/单一值处理

    输入：
    dataForPreprocess: 预处理前的数据集 DataFrame
    listDropRepeat: 确认数据集是否重复的字段列表 list
    listDropSingle: 需要检查单一值的字段列表 list
    thresholdSingleDrop: 单一值处理阀值 float

    输出：
    函数repeat_drop_value: 删除重复值、空行
    函数single_drop_value: 单一值处理函数

    管理记录：
    1. edited by 王文丹 2021/07/02 

    """
    def __init__(self,thresholdSingleDrop):

        self.thresholdSingleDrop = thresholdSingleDrop  #单一值处理阀值 float

    def repeat_drop_value(self,dataForPreprocess,listDropRepeat):
    
        """
        功能描述：删除重复值、空行

        输入：
        dataForPreprocess: 预处理前的数据集 DataFrame
        listDropRepeat: 确认数据集是否重复的字段列表 list

        输出：
        dataRepeatDele: 重复值和空行处理后的数据集 DataFrame

        管理记录：
        1. edited by 王文丹 2021/07/02
        
        """
        try:

            dataRepeatDele = dataForPreprocess.drop_duplicates(listDropRepeat) # 删除重复行
            dataRepeatDele = dataRepeatDele.dropna(axis=0,how='all') # 删除空行

            return dataRepeatDele

        except Exception as e:

            print('C_PreprocessData,RepeatValueModule,RepeatValueHandle,RepeatValueClass,repeat_drop_value',':',e)

    def single_drop_value(self,dataForPreprocess,listDropSingle):
    
        """
        功能描述：单一值处理函数

        输入：
        dataForPreprocess: 预处理前的数据集 DataFrame
        listDropSingle: 需要检查单一值的字段列表 list
        thresholdSingleDrop: 单一值处理阀值 float

        输出：
        colSingleDele: 由于单一值太高被删除的特征列表 list
        dataRepeatDele: 单一值处理后的数据集 DataFrame

        管理记录：
        1. edited by 王文丹 2021/07/02
        
        """
        try:

            colSingleDele =[]
            for col_constant in listDropSingle:
                if dataForPreprocess[col_constant].value_counts(dropna=False,normalize=True).max()>self.thresholdSingleDrop:
                    colSingleDele.append(col_constant)
                    
            dataSingleDele = dataForPreprocess.drop(columns=colSingleDele)
        
        except Exception as e:

            print('C_PreprocessData,RepeatValueModule,RepeatValueHandle,RepeatValueClass,single_drop_value',':',e)
       
        return (colSingleDele,dataSingleDele)