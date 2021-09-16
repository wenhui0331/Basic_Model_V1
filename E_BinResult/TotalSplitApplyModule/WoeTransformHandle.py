import pandas as pd
import numpy as np
from D_SplitData.SplitPointApplyModule.SplitPointApplyHandle import SplitPointApplyClass

class WoeTransformClass():

    """
    功能描述：挑选IV高的变量对其进行WOE变化
            （尽量不要让所有变量都参与WOE变换，否则会使得数据存储量过大）
             

    输入：
    dataForSelectBin: 用于分箱的数据集合 DataFrame
    regroupSplitedDictSort： 记录分箱完成后的统计性描述groupby结果(按IV值从大到小排序) Dict 内含IV列
    WOEBinDictSort: 记录分箱完成后的WOE映射结果(按IV值从大到小排序) Dict
    ivThreshold: IV筛选阈值 float

    输出：
    函数woe_transform：挑选IV高的变量对其进行WOE变化，并挑选出高IV值变量

    管理记录：
    1. edited by 王文丹 2021/07/25
    
    """

    def __init__(self, specialValue, ylabel):

        self.specialValue = specialValue # 特殊值 如'missing'或'-999999'
        self.ylabel = ylabel             # y标签的特征名称

    def woe_transform(self,dataForSelectBin,regroupSplitedDictSort,WOEBinDictSort,ivThreshold):
        """
        功能描述：挑选IV高的变量对其进行WOE变化
                （尽量不要让所有变量都参与WOE变换，否则会使得数据存储量过大）
        
        输入：
        dataForSelectBin: 用于分箱的数据集合,该数据已经完成了分箱Bin分配 DataFrame 
        regroupSplitedDictSort： 记录分箱完成后的统计性描述groupby结果(按IV值从大到小排序) Dict 内含IV列
        WOEBinDictSort: 记录分箱完成后的WOE映射结果(按IV值从大到小排序) Dict
        ivThreshold: IV筛选阈值 float

        输出
        dataForSelectBin: 对IV值高的变量，已增加WOE转换的数据列（IV值低的变量未进行转换） DataFrame
        ivHighVarBinName：高IV值变量集合, 格式为['col1_Bin,'col2_Bin'] list
        ivLowVarBinName: 低IV值变量集合, 格式为['col11_Bin,'col21_Bin'] list
        ivHighVarWOEName：高IV值变量集合, 格式为['col1_Bin_WOE','col2_Bin_WOE'] list
        ivHighDict：高IV值 IV映射字典 格式为{'d1_Bin': 3, 'd2_Bin': 6}

        管理记录：
        1. edited by 王文丹 2021/07/25

        """
        try:
          
            # 针对已分箱，已计算的统计性描述结果，计算IV值，且筛选高于阈值的变量
            # ivHigh格式为[('d1_Bin', 0.03), ('d2_Bin', 0.06)]  
            # 此处的regroupSplitedDictSort字段带Bin {'现有贷款授信银行数_Bin':DataFrame1,'现有贷款授信银行数1_Bin':DataFrame1}
            ivHigh = [(j,k['IV'].sum()) for j,k in regroupSplitedDictSort.items() if k['IV'].sum() >= ivThreshold]
            ivLow = [(j,k['IV'].sum()) for j,k in regroupSplitedDictSort.items() if k['IV'].sum() < ivThreshold]

            # 针对已分箱，已计算的统计性描述结果，计算IV值，且筛选高于阈值的变量
            # ivHigh格式为['d1_Bin': 0.03, 'd2_Bin': 0.06]
            ivHighDict = {j:k['IV'].sum() for j,k in regroupSplitedDictSort.items() if k['IV'].sum() >= ivThreshold}

            # 按IV值高低对变量进行排序
            # 格式为[('d2_Bin', 6), ('d1_Bin', 3)]
            ivHighSorted = sorted(ivHigh,key=lambda x:x[1],reverse=True)
            ivLowSorted = sorted(ivLow,key=lambda x:x[1],reverse=True)

            # 存储高IV值的变量
            ivHighVarBinName = [i[0] for i in ivHighSorted]
            ivLowVarBinName = [i[0] for i in ivLowSorted]
            ivHighVarWOEName = []

            for var in ivHighVarBinName:
                woeVar = var+'_WOE'  # 此处的var必须为_Bin结尾
                ivHighVarWOEName.append(woeVar)
                dataForSelectBin[woeVar] = dataForSelectBin[var].map(lambda x: WOEBinDictSort[var][x]['WOE'])
            
            return (dataForSelectBin,ivHighVarBinName,ivLowVarBinName,ivHighVarWOEName,ivHighDict)
        
        except Exception as e:
            print('E_BinResult,TotalSplitApplyModule,WoeTransformHandle,woe_transform,woe_transform,',':',e)

 

    def apply_splitPoint_to_test(self,dataForSelectTest,colListForTransform,trainSplitedDictSort,trainWOEBinDictSort):
        """
        功能描述：将训练集合的分箱切点结果运用至测试集合上

        输入：
        dataForSelectTest: 测试集合
        colListForTransform: 需要切点运用的特征列表，此书已经为Bin格式（一般是经过IV值挑选后的变量列表，如上述函数中的ivHighVarBinName）
                            如['现有贷款授信银行数_Bin', '有信用业务的银行数_Bin']
        trainSplitedDictSort: 训练集合输出的分箱切点 Dict
        trainWOEBinDictSort: 训练集合输出的WOE映射关系 Dict

        输出：
        dataForSelectTest: 已进行Bin和WOE映射的测试集合

        管理记录：
        1. edited by 王文丹 2021/07/25
        """

        try:

            for var in colListForTransform:
                # 对切分点运用类进行重命名
                SPAC = SplitPointApplyClass(self.specialValue,self.ylabel)
                splitedPoints = trainSplitedDictSort[var]
                # 给测试集合分配切点
                dataForSelectTest[var] = SPAC.split_Bin(dataForSelectTest, var.replace('_Bin',''), splitedPoints)
                # 给训练集合分配WOE
                dataForSelectTest[var+'_WOE'] = dataForSelectTest[var].map(lambda x: trainWOEBinDictSort[var][x]['WOE'])
                
            return dataForSelectTest

        except Exception as e:
            print('E_BinResult,TotalSplitApplyModule,WoeTransformHandle,woe_transform,apply_splitPoint_to_test,',':',e)

            






            

            











