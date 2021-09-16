from statsmodels.api import Logit
import numpy as np
import pandas as pd

class LinearFitWithScoreScaleClass():

    """
    功能描述：分数的线性拟合和分数刻度设置

    输入：
    dataForLrTrain: 以备线性拟合的WOE数据集 DataFrame
    fiteredvarWoe: 完成特征筛选的特征名称（WOE转换过）list
    proda: 违约概率，为筛选后WOE变量的线性拟合汇总数值
    baseScore: 基础分数
    pdo: 分数刻度

    输出：
    函数proba_2Score：根据违约概率、基础分数、分数刻度，输出评分卡分数 
    函数linear_fit: 针对已经挑选好的变量，进行线性拟合并输出评分卡分数

    管理记录：
    1. edited by 王文丹 2021/07/26
    """

    def __init__(self,ylabel,baseScore,pdo):

        self.ylabel = ylabel  #好坏标签的特征名称 如'y'
        self.baseScore = baseScore #基础分数
        self.pdo = pdo #分数刻度
    
    def proba_2Score(self,proba):
        """
        功能描述：根据违约概率、基础分数、分数刻度，输出评分卡分数

        输入：
        proda: 违约概率，为筛选后WOE变量的线性拟合汇总数值
        baseScore: 基础分数
        pdo: 分数刻度

        输出：
        score: 模型分数

        管理记录：
        1. edited by 王文丹 2021/07/26
        """
        odds=proba/(1-proba)
        score=self.baseScore-self.pdo/np.log(2)*np.log(odds)
        return score

    def linear_fit(self,dataForLr,fiteredvarWoe):

        """
        功能描述：针对已经挑选好的变量，进行线性拟合并输出评分卡分数

        输入：
        dataForLrTrain: 以备线性拟合的WOE数据集 DataFrame
        fiteredvarWoe: 完成特征筛选的特征名称（WOE转换过）list

        输出：
        dataForLr: 带违约概率和模型分数的数据集合 DataFrame
        lrSmallP: 线性拟合的所有汇总性结果

        管理记录：
        1. edited by 王文丹 2021/07/26
        """

        xSmallP = dataForLr[fiteredvarWoe]
        xSmallP['intercept'] = [1]*xSmallP.shape[0]
        print(xSmallP)
        lrSmallP = Logit(dataForLr[self.ylabel],xSmallP).fit()
        lrSmallPSummary = lrSmallP.summary()
        print(lrSmallPSummary) # 输出所有拟合的汇总性结果
        dataForLr['yPred'] = lrSmallP.predict(xSmallP)
        dataForLr['score'] = dataForLr['yPred'].map(lambda x:self.proba_2Score(x))

        return (dataForLr,lrSmallPSummary)

    





        