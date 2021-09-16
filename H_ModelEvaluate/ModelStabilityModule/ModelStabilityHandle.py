import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, roc_auc_score
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


class ModelStabilityClass(object):

    """
    功能： 评分卡稳定性指标计算
    

    输入：
    dataInitial: 数据集 dataframe 
    flagName: 标签名称 string 
    factorName: 指标名称 string 
    cutMethod: 分数切分方式，等频或者等距，如：'step', 'quantile' 
    nBin: 切分区间个数 int
    cutPoint: 指定切点，包含最大最小值 list，如：[-np.inf, 20, 30, 50, np.inf]
    

    输出：
    函数 cut_bin_df: 为数据集指定指标等频、等距或者指定切点分组
    函数 get_psi：计算psi，并返回psi分组详情


    维护记录：
    1. created by hean 2021/07/07


    """



    def __init__(self):

        """
        功能：评分卡稳定性指标计算-类ModelStabilityClass的初始化函数

        输入：
        dataBase: 基准数据集 dataframe 
        dataInitial: 比较数据集 dataframe 
        flagName: 标签名称 string 
        factorName: 指标名称 string 
        cutMethod: 分数切分方式，等频或者等距，如：'step', 'quantile' 
        nBin: 切分区间个数 int
        cutPoint: 指定切点，包含最大最小值 list，如：[-np.inf, 20, 30, 50, np.inf]
        
        输出：无
        
        管理记录：
        1. created by hean 2021/07/07

        """
        pass


    def cut_bin_df(self, dataInitial, flagName, factorName, cutMethod='step', nBin=20, cutPoint=[]):

        '''
        功能：为数据集指定指标等频、等距或者指定切点分组

        输入：
        dataInitial: 数据集 dataframe 
        flagName: 标签名称 string 
        factorName: 指标名称 string 
        cutMethod: 分数切分方式，等频或者等距，如：'step', 'quantile' 
        nBin: 切分区间个数 int
        cutPoint: 指定切点，包含最大最小值 list，如：[-np.inf, 20, 30, 50, np.inf]

        输出：
        resultDf: 为数据集指定指标等频、等距或者指定切点分组 dataframe  
        binRange: 分组切点列表 list
        
        管理记录：
        1. created by hean 2021/07/07
        
        '''

        try:
        
            if len(dataInitial) == 0:
                return pd.DataFrame(), []
            
            dataInitial = dataInitial[[flagName, factorName]].copy()
            if len(cutPoint) > 0:
                binRange = cutPoint
            else:
                if cutMethod=='step':
                    _, BincutPoint = pd.cut(dataInitial[factorName], nBin, retbins=True)
                else:
                    _, BincutPoint = pd.qcut(dataInitial[factorName], nBin, retbins=True)
                binRange = sorted(list(set([round(x) for x in BincutPoint])))
                binRange[0], binRange[-1] = -np.inf, np.inf
            
            dataInitial[factorName+'_bin'] = pd.cut(dataInitial[factorName], binRange).cat.add_categories(
                    ["missing"]).fillna("missing").cat.remove_unused_categories()

            return dataInitial, binRange
        
        except Exception as e:
            print('H_ModelEvaluateData,ModelStabilityClass,cut_bin_df',':',e) 
    
    
    def get_psi(self, dataBase, dataInitial, flagName, factorName, cutMethod='step', nBin=20, cutPoint=[]):

        '''
        功能：计算psi

        输入：
        dataBase: 基准数据集 dataframe 
        dataInitial: 比较数据集 dataframe 
        flagName: 标签名称 string 
        factorName: 指标名称 string 
        cutMethod: 分数切分方式，等频或者等距或者不切分，如：'step', 'quantile', 'none'
        nBin: 切分区间个数 int
        cutPoint: 指定切点，包含最大最小值 list，如：[-np.inf, 20, 30, 50, np.inf]

        输出：
        psi: psi值 float  
        psiDf: psi矩阵 dataframe
        
        管理记录：
        1. created by hean 2021/07/07
        
        '''

        try:
        
            if cutMethod=='none':
                factorName = factorName
            else:
                dataBase,cutPoint = self.cut_bin_df(dataBase, flagName, factorName, cutMethod, nBin, cutPoint)
                dataInitial,cutPoint = self.cut_bin_df(dataInitial, flagName, factorName, cutMethod, nBin, cutPoint)
                factorName = factorName+'_bin'

            RegroupOne = dataBase.groupby(by=factorName)[flagName].count()
            DevDf = pd.DataFrame({'dev': RegroupOne})
            RegroupTwo = dataInitial.groupby(by=factorName)[flagName].count()
            ValDf = pd.DataFrame({'val': RegroupTwo})
            psiDf = pd.merge(DevDf, ValDf, left_index=True, right_index=True)
            psiDf['%dev'] = psiDf['dev'] / psiDf['dev'].sum()
            psiDf['%val'] = psiDf['val'] / psiDf['val'].sum()
            psiDf['psi'] = psiDf.apply(lambda x: (x['%val']-x['%dev'])*np.log(x['%val']/x['%dev']) if x['%val']!=0 else 0, axis=1)
            psi = psiDf['psi'].sum()
            
            return psi, psiDf
        
        except Exception as e:
            print('H_ModelEvaluateData,ModelStabilityClass,get_psi',':',e) 