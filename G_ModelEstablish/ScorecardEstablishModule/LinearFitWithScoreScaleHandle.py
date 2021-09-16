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
        lrSmallPResult: 线性拟合的所有汇总性pandas提取

        管理记录：
        1. edited by 王文丹 2021/07/26
        """

        xSmallP = dataForLr[fiteredvarWoe]
        xSmallP['intercept'] = [1]*xSmallP.shape[0]
        # print(xSmallP)
        lrSmallP = Logit(dataForLr[self.ylabel],xSmallP).fit()
        lrSmallPSummary = lrSmallP.summary()
        print(lrSmallPSummary) # 输出所有拟合的汇总性结果
        dataForLr['yPred'] = lrSmallP.predict(xSmallP)
        dataForLr['score'] = dataForLr['yPred'].map(lambda x:self.proba_2Score(x))

        # 保存Logit逻辑回归的结果参数pandas格式
        lrSmallPResultLeft = pd.DataFrame([lrSmallP.params,lrSmallP.bse,lrSmallP.tvalues,lrSmallP.pvalues]).T.reset_index(level=0)
        lrSmallPResultRight = pd.DataFrame(lrSmallP.conf_int()).reset_index(level=0)
        lrSmallPResult = pd.merge(lrSmallPResultLeft,lrSmallPResultRight,on='index',how = 'left')
        lrSmallPResult.columns = ['特征名称','coef','std err','z','p>|z|','[0.025','0.975]']

        # 分数打分表
        ScoreFrame1 = -(self.pdo/np.log(2))*(list(lrSmallP.params)*xSmallP)
        xSmallP.columns = [col.replace('intercept','intercept_Bin_WOE') for col in xSmallP.columns]
        ScoreFrame1.columns = [col.replace('_Bin_WOE','')+'_Score' for col in ScoreFrame1.columns]
        ScoreFrame = pd.concat([xSmallP,ScoreFrame1],axis=1)
        
        # 分数分组打分Regroup
        ScoreRegroup = {}
        # colScoreName = [col.replace('_Score','') for col in ScoreFrame1.columns]
        for col in ScoreFrame1.columns:
            col = col.replace('_Score','')
            SingleScoreBin = sorted(set(ScoreFrame[col+'_Bin_WOE']))
            n1 = pd.DataFrame(SingleScoreBin)
            n1['分数分档'] = np.inf
            n1.columns = [col+'_Bin_WOE','分数分档']
            for i in SingleScoreBin:
                n1.loc[n1[col+'_Bin_WOE']==i,'分数分档'] = ScoreFrame[ScoreFrame[col+'_Bin_WOE']==i].reset_index(level=0)[col+'_Score'][0]
            ScoreRegroup[col] = n1

        return (dataForLr,lrSmallPSummary,lrSmallPResult,ScoreFrame,ScoreRegroup)

    





        