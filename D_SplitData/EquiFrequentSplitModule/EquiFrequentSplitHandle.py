### 等频分箱
import pandas as pd
import numpy as np
from D_SplitData.SplitPointApplyModule.SplitPointApplyHandle import SplitPointApplyClass

class EquiFrequentSplitClass():

    """
    功能描述：输出等频分箱切点

    输入：
    dataForSplit: 经过数据清洗后的数据集 DataFrame
    colForSplit: 用于分箱的特征 str
    numForSplit: 分箱数目 一般为50或100 int
    specialValue: 特殊值 如'missing' 或 '-999999'
    ylabel：坏样本率的特征名称 如'y' str
    singleBadRate01: 单箱全为坏样本或好样本的合并参数 0：不需要合并 1：需要合并 int

    输出：
    函数equi_freuent_split：输出等频分箱切点的函数

    管理记录：
    1. edited by 王文丹 2021/07/09 

    """

    def __init__(self,numForSplit,specialValue,ylabel,singleBadRate01):

        self.numForSplit = numForSplit   # 分箱数目 int
        self.specialValue = specialValue # 特殊值 如'missing'或'-999999'
        self.ylabel = ylabel   # 坏样本率的特征名称 如'y'
        self.singleBadRate01 = singleBadRate01 # 单箱占比为0的合并参数 0：不需要合并 1：需要合并

    def equi_freuent_split(self,dataForSplit,colForSplit):

        """
        功能描述：输出等频分箱切点

        输入：
        dataForSplit： 经过数据清洗后的数据集 DataFrame
        colForSplit：用于分箱的特征 char
        numForSplit：分箱数目 int
        specialValue：特殊值（如因缺失造成的另分一箱） 如'missing'或'-999999'

        输出：
        SplitedPoint：等距分箱的切分点 list

        管理记录：
        1. edited by 王文丹 2021/07/09
        
        """

        try:

            # 剔除 特殊值 比如因缺失造成的另分一箱
            dataForSplitNoSpecial = dataForSplit[dataForSplit[colForSplit]!=self.specialValue]
            # 进行列中不同值的判断
            fisrtCut = sorted(list(set(dataForSplitNoSpecial[colForSplit])))
            if len(fisrtCut) <= 1:
                SplitedPoint = fisrtCut
                return SplitedPoint
            
            
            # 等频分箱语句
            (results,bin_edags) = pd.qcut(dataForSplitNoSpecial[colForSplit],q=self.numForSplit,retbins=True,precision=3, duplicates='drop')
            SplitedPoint = list(np.round(bin_edags,4))
            if len(SplitedPoint) > 1:
                SplitedPoint.remove(SplitedPoint[-1])
            SplitedPoint = sorted(list(set((SplitedPoint))))

            # 对切分点运用类进行重命名
            SPAC = SplitPointApplyClass(self.specialValue,self.ylabel)
            # 分配切点
            dataForSplitNoSpecial.loc[:,colForSplit+'_Bin'] = SPAC.split_Bin(dataForSplitNoSpecial, colForSplit, SplitedPoint)   
            # 形成初始切割点的分箱描述性结果
            splitedBG = SPAC.split_result_bg(dataForSplitNoSpecial,colForSplit+'_Bin',SplitedPoint)  
            # print(splitedBG)

            # 特殊处理1：循环进行切点合并，使得每个分箱不全为好样本或者坏样本
            # 特殊处理1：循环进行切点合并，使得每个分箱不全为好样本或者坏样本
            # 特殊处理1：循环进行切点合并，使得每个分箱不全为好样本或者坏样本
            splitedBG = splitedBG[splitedBG[colForSplit+'_Bin']!='missing']
            indexGoodZeros = list(splitedBG[splitedBG['good']==0].index)
            indexBadZeros = list(splitedBG[splitedBG['bad']==0].index)
            indexAllNull = list(splitedBG[splitedBG['All'].isnull()].index)
            indexOfZeros = sorted(list(set(indexGoodZeros + indexBadZeros + indexAllNull)))
            # indexOfZeros = list((splitedBG[splitedBG['bad']==0].index) | (splitedBG[splitedBG['good']==0].index))
            while len(indexOfZeros) > 0 and self.singleBadRate01 == 1 and len(SplitedPoint) > 1: 
                SplitedPoint = SPAC.combine_zero(splitedBG, SplitedPoint,colForSplit+'_Bin')
                # 重新分配切点
                dataForSplitNoSpecial[colForSplit+'_Bin'] = SPAC.split_Bin(dataForSplitNoSpecial, colForSplit, SplitedPoint)   
                # 重新形成描述性结果        
                splitedBG = SPAC.split_result_bg(dataForSplitNoSpecial, colForSplit+'_Bin', SplitedPoint)  
                splitedBG = splitedBG[splitedBG[colForSplit+'_Bin']!='missing']
                indexGoodZeros = list(splitedBG[splitedBG['good']==0].index)
                indexBadZeros = list(splitedBG[splitedBG['bad']==0].index)
                indexAllNull = list(splitedBG[splitedBG['All'].isnull()].index)
                indexOfZeros = sorted(list(set(indexGoodZeros + indexBadZeros + indexAllNull)))
                # indexOfZeros = list((splitedBG[splitedBG['bad']==0].index) | (splitedBG[splitedBG['good']==0].index))

            return SplitedPoint

        except Exception as e:
            print('D_SplitData,EquiDistanceSplitModule,EquiDistanceSplitHandle,EquiFrequentSplitClass,equi_freuent_split',':',e)



        









