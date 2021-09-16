import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, roc_auc_score
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from ..ModelStabilityModule.ModelStabilityHandle import ModelStabilityClass


class ModelEffectClass(object):

    """
    功能： 评分卡评价指标计算
    

    输入：
    dataInitial: 数据集 dataframe 
    flagName: 标签名称 string 
    factorName: 指标名称 string 
    cutMethod: 分数切分方式，等频或者等距，如：'step', 'quantile' 
    nBin: 切分区间个数 int
    cutPoint: 指定切点，包含最大最小值 list，如：[-np.inf, 20, 30, 50, np.inf]
    

    输出：
    函数 group_by_df：转换数据格式，变为指标每一分区下好样本的个数、坏样本的个数、ks值、lift 等
    函数 get_ks：计算KS
    函数 get_auc：计算AUC
    函数 get_gini：计算gini
    函数 get_report：评分卡各项评价指标汇总报告
    函数 plot_ks：绘制KS曲线，并保存图片文件
    函数 plot_lift：绘制lift曲线，并保存图片文件
    函数 plot_pr：绘制PR曲线，并保存图片文件
    函数 plot_roc：绘制ROC曲线，并保存图片文件
    函数 plot_pr_f1：绘制召回率、精确率、f1-score曲线图，并保存图片文件
    函数 plot_ks_distribution：绘制累计好坏客户的曲线图，和分数分布曲线，并保存图片文件
    函数 plot_good_bad_ind：绘制好客户、坏客户、中间客户分数分布图，并保存图片文件
    函数 plot_cumulative_approve：绘制累计好坏通过率图，并保存图片文件


    维护记录：
    1. created by hean  2021/07/07


    """



    def __init__(self):

        """
        功能：评分卡评价指标计算-类ModelEffectClass的初始化函数

        输入：
        dataInitial: 数据集 dataframe 
        flagName: 标签名称 string 
        factorName: 指标名称 string 
        
        输出：无
        
        管理记录：
        1. created by hean 2021/07/07

        """

        pass


    def group_by_df(self, dataInitial, flagName, factorName):

        '''
        功能：转换数据格式，变为指标每一分区下好样本的个数、坏样本的个数、ks值、lift等

        输入：
        dataInitial: 数据集 dataframe 
        flagName: 标签名称 string 
        factorName: 指标名称 string 

        输出：
        resultDf: 指标每一分区下好样本的个数、坏样本的个数、ks值、lift等 dataframe  
        
        管理记录：
        1. created by hean 2021/07/07
        
        '''

        try:
        
            if len(dataInitial) == 0:
                return pd.DataFrame()
            
            dataInitial = dataInitial[[flagName, factorName]].copy()
            
            RegroupOne = dataInitial.groupby([factorName])[flagName].count()
            RegroupTwo = dataInitial.groupby([factorName])[flagName].sum()
            resultDf = pd.DataFrame({'good': RegroupOne-RegroupTwo, 'bad': RegroupTwo}).reset_index()
            resultDf['total'] = resultDf['good'] + resultDf['bad']
            resultDf['%total'] = resultDf['total'].cumsum() / resultDf['total'].sum()
            resultDf['%Bad_Rate'] = resultDf['bad']/(resultDf['bad']+resultDf['good'])
            resultDf = resultDf.sort_values(by=[factorName], ascending=True)
            resultDf = resultDf.reset_index(drop=True)
            resultDf['cumsum_bad'] = resultDf['bad'].cumsum() / resultDf['bad'].sum()
            resultDf['cumsum_good'] = resultDf['good'].cumsum() / resultDf['good'].sum()
            resultDf['ks'] = resultDf['cumsum_bad'] - resultDf['cumsum_good']
            resultDf['lift'] = (resultDf['bad'].cumsum()/resultDf['total'].cumsum()) / (resultDf['bad'].sum()/resultDf['total'].sum())
            resultDf['accuracy'] = (resultDf['bad'].cumsum()+resultDf['good'].sum()-resultDf['good'].cumsum())/resultDf['total'].sum()
            resultDf['precision'] = resultDf['bad'].cumsum() / (resultDf['bad'].cumsum()+resultDf['good'].cumsum())
            resultDf['recall'] = resultDf['bad'].cumsum() / resultDf['bad'].sum()
            resultDf['specificity'] = 1 - resultDf['good'].cumsum() / resultDf['good'].sum()
            resultDf['FPR'] = resultDf['good'].cumsum() / resultDf['good'].sum()
            resultDf['FNR'] = 1 - resultDf['bad'].cumsum() / resultDf['bad'].sum()
            resultDf['F1_score'] = 2 / (1/resultDf['precision'] + 1/resultDf['recall'])
            return resultDf
        
        except Exception as e:
            print('H_ModelEvaluateData,ModelEffectClass,group_by_df',':',e) 


    def get_ks(self, dataInitial, flagName, factorName):

        '''
        功能：计算KS

        输入：
        dataInitial: 数据集 dataframe 
        flagName: 标签名称 string 
        factorName: 指标名称 string 

        输出：
        maxKs: ks值 float  
        
        管理记录：
        1. created by hean 2021/07/07
        
        '''

        try:
        
            resultDf = self.group_by_df(dataInitial, flagName, factorName)
            maxKs = max(resultDf['ks'])
            
            return maxKs
        
        except Exception as e:
            print('H_ModelEvaluateData,ModelEffectClass,get_ks',':',e) 


    def get_auc(self, dataInitial, flagName, factorName):

        '''
        功能：计算AUC

        输入：
        dataInitial: 数据集 dataframe 
        flagName: 标签名称 string 
        factorName: 指标名称 string 

        输出：
        auc: auc值 float  
        
        管理记录：
        1. created by hean 2021/07/07
        
        '''

        try:
        
            dataInitial[factorName].astype(float)
            dataInitial = dataInitial[dataInitial[factorName].notnull()]
            fpr, tpr, _ = roc_curve(dataInitial[flagName], -1*dataInitial[factorName])
            Auc = auc(fpr, tpr)
            
            return Auc
        
        except Exception as e:
            print('H_ModelEvaluateData,ModelEffectClass,get_auc',':',e) 


    def get_gini(self, dataInitial, flagName, factorName):

        '''
        功能：计算gini

        输入：
        dataInitial: 数据集 dataframe 
        flagName: 标签名称 string 
        factorName: 指标名称 string 

        输出：
        gini: gini系数 float  
        
        管理记录：
        1. created by hean 2021/07/07
        
        '''

        try:
        
            auc = self.get_auc(dataInitial, flagName, factorName)
            gini = 2*auc - 1
            return gini
        
        except Exception as e:
            print('H_ModelEvaluateData,ModelEffectClass,get_gini',':',e) 


    def get_report(self, dataInitial, flagName, factorName, cutMethod='step', nBin=20, cutPoint=[]):

        '''
        功能：评分卡各项评价指标汇总报告

        输入：
        dataInitial: 数据集 dataframe 
        flagName: 标签名称 string 
        factorName: 指标名称 string 
        cutMethod: 分数切分方式，等频或者等距或者不切分，如：'step', 'quantile', 'none'
        nBin: 切分区间个数 int
        cutPoint: 指定切点，包含最大最小值 list，如：[-np.inf, 20, 30, 50, np.inf]

        输出：
        df: 模型评价报告 dataframe 
        
        管理记录：
        1. created by hean 2021/07/07
         
        '''

        try:
        
            if cutMethod=='none':
                factorName = factorName
            else:
                dataInitial,_ = ModelStabilityClass.cut_bin_df(self, dataInitial,flagName,factorName,cutMethod,nBin,cutPoint)
                factorName = factorName+'_bin'
            
            df = self.group_by_df(dataInitial,flagName,factorName)
            
            return df
        
        except Exception as e:
            print('H_ModelEvaluateData,ModelEffectClass,get_report',':',e) 


    def plot_ks(self, dataInitial, flagName, factorName, dfName, imagePath):

        '''
        功能：绘制KS曲线

        输入：
        dataInitial: 数据集 dataframe 
        flagName: 标签名称 string 
        factorName: 指标名称 string 
        dfName: 集合名称 string
        imagePath: 图片保存路径 string 

        输出：
        KS曲线图，文件
        
        管理记录：
        1. created by hean 2021/07/07
        
        '''

        try:
        
            resultDf = self.group_by_df(dataInitial, flagName, factorName)
            ks = max(resultDf['ks'])
            KsIndex=resultDf[resultDf['ks']==ks].index[0]
            KsValue=resultDf[factorName][KsIndex]
            
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            sns.set(font='SimHei', style='white')
            fig = plt.figure(figsize=(12,7))
            ax1 = fig.add_subplot(111)
            ax1.patch.set_facecolor("gainsboro")
            title="KS Curve of %s\nKS=%.2f , KS_cutoff=%s"%(dfName, ks, KsValue)
            ax1.plot(resultDf[factorName], resultDf['cumsum_good'], color='g', label='Cumsum_Good')
            ax1.plot(resultDf[factorName], resultDf['cumsum_bad'], color='r', label='Cumsum_Bad')
            ax1.plot(resultDf[factorName], resultDf['ks'], color='b', label='KS', linestyle='--')
            ax1.set_xlabel('Score')
            ax1.set_ylabel('%Cumulative_Rate')
            ax1.legend()
            ax1.set_title(title)
            plt.rcParams['savefig.facecolor'] = 'whitesmoke'
            plt.savefig(imagePath+'/%s_KS.png'%dfName)
            plt.show()
        
        except Exception as e:
            print('H_ModelEvaluateData,ModelEffectClass,plot_ks',':',e) 


    def plot_lift(self,dataInitial,dataVal,dataOot,flagName,factorName,imagePath):

        '''
        功能：绘制lift曲线

        输入：
        dataDev: 训练数据集 dataframe 
        dataVal: 测试数据集 dataframe 
        dataOot: 时间外数据集 dataframe （如果没有，则填 []）
        flagName: 标签名称 string 
        factorName: 指标名称 string 
        imagePath: 图片保存路径 string

        输出：
        lift曲线图，文件
        
        管理记录：
        1. created by hean 2021/07/07
        
        '''

        try:
        
            dataInitial,cutPoint = ModelStabilityClass.cut_bin_df(self, dataInitial,flagName,factorName,'quantile',20)
            dataVal,_ = ModelStabilityClass.cut_bin_df(self, dataVal,flagName,factorName,'quantile',20,cutPoint)
            factorNameNew = factorName+'_bin'
            DfDev = self.group_by_df(dataInitial, flagName, factorNameNew)
            DfVal = self.group_by_df(dataVal, flagName, factorNameNew)
            DevLift = DfDev[[factorNameNew, 'lift']].copy()
            ValLift = DfVal[[factorNameNew, 'lift']].copy()
            DevLift['type'] = 'dev'
            ValLift['type'] = 'val'
            DfLift = pd.concat([DevLift, ValLift])
            title="LIFT Curve"
            
            if len(dataOot)>0:
                dataOot,_ = ModelStabilityClass.cut_bin_df(self, dataOot,flagName,factorName,'quantile',20,cutPoint)
                DfOot = self.group_by_df(dataOot, flagName, factorNameNew)
                OotLift = DfOot[[factorNameNew, 'lift']].copy()
                OotLift['type'] = 'oot'
                DfLift = pd.concat([DfLift, OotLift])
            
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            sns.set(font='SimHei', style='white')
            fig = plt.figure(figsize=(12,7))
            ax1 = fig.add_subplot(111)
            ax1.patch.set_facecolor("gainsboro")
            for i in DfLift.groupby('type'):
                ax1.plot(i[1].index, i[1]['lift'], marker='o', label=i[0])
            ax1.set_xlabel('score bin of data')
            ax1.set_ylabel('Lift')
            ax1.set_xticklabels(i[1][factorNameNew][::2], rotation=15)
            ax1.legend()
            ax1.set_title(title)
            plt.rcParams['savefig.facecolor'] = 'whitesmoke'
            plt.savefig(imagePath+'/LIFT.png')
        
        except Exception as e:
            print('H_ModelEvaluateData,ModelEffectClass,plot_lift',':',e) 


    def plot_pr(self,dataDev,dataVal,dataOot,flagName,factorName,imagePath):

        '''
        功能：绘制PR曲线

        输入：
        dataDev: 训练数据集 dataframe 
        dataVal: 测试数据集 dataframe 
        dataOot: 时间外数据集 dataframe （如果没有，则填 []）
        flagName: 标签名称 string 
        factorName: 指标名称 string 
        imagePath: 图片保存路径 string

        输出：
        PR曲线图，文件
        
        管理记录：
        1. created by hean 2021/07/07
        
        '''

        try:
        
            DfDev = self.group_by_df(dataDev, flagName, factorName)
            DfVal = self.group_by_df(dataVal, flagName, factorName)
            DevPr = DfDev[['precision', 'recall']].copy()
            ValPr = DfVal[['precision', 'recall']].copy()
            DevPr['type'] = 'dev'
            ValPr['type'] = 'val'
            DfPr = pd.concat([DevPr, ValPr])
            title="Precision-Recall Curve"
            
            if len(dataOot)>0:
                DfOot = self.group_by_df(dataOot, flagName, factorName)
                OotPr = DfOot[['precision', 'recall']].copy()
                OotPr['type'] = 'oot'
                DfPr = pd.concat([DfPr, OotPr])
            
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            sns.set(font='SimHei', style='white')
            fig = plt.figure(figsize=(12,7))
            ax1 = fig.add_subplot(111)
            ax1.patch.set_facecolor("gainsboro")
            for i in DfPr.groupby('type'):
                ax1.plot(i[1]['recall'], i[1]['precision'], label=i[0])
            ax1.set_xlabel('Recall')
            ax1.set_ylabel('Precision')
            ax1.legend()
            ax1.set_title(title)
            plt.rcParams['savefig.facecolor'] = 'whitesmoke'
            plt.savefig(imagePath+'/PR.png')
            plt.show()
        
        except Exception as e:
            print('H_ModelEvaluateData,ModelEffectClass,plot_pr',':',e) 


    def plot_roc(self,dataDev,dataVal,dataOot,flagName,factorName,imagePath):

        '''
        功能：绘制ROC曲线

        输入：
        dataDev: 训练数据集 dataframe 
        dataVal: 测试数据集 dataframe 
        dataOot: 时间外数据集 dataframe （如果没有，则填 []）
        flagName: 标签名称 string 
        factorName: 指标名称 string 
        imagePath: 图片保存路径 string

        输出：
        ROC曲线图，文件
        
        管理记录：
        1. created by hean 2021/07/07
        
        '''

        try:
        
            DfDev = self.group_by_df(dataDev, flagName, factorName)
            DfVal = self.group_by_df(dataVal, flagName, factorName)
            DevRoc = DfDev[['FPR', 'recall']].copy()
            ValRoc = DfVal[['FPR', 'recall']].copy()
            AucDev = roc_auc_score(dataDev[flagName], -1*dataDev[factorName])
            AucVal = roc_auc_score(dataVal[flagName], -1*dataVal[factorName])
            DevRoc['type'] = 'auc of dev: {}'.format(round(AucDev,4))
            ValRoc['type'] = 'auc of val: {}'.format(round(AucVal,4))
            DfRoc = pd.concat([DevRoc, ValRoc])
            title="ROC Curve"
            
            if len(dataOot)>0:
                DfOot = self.group_by_df(dataOot, flagName, factorName)
                OotRoc = DfOot[['FPR', 'recall']].copy()
                AucOot = roc_auc_score(dataOot[flagName], -1*dataOot[factorName])
                OotRoc['type'] = 'auc of oot: {}'.format(round(AucOot,4))
                DfRoc = pd.concat([DfRoc, OotRoc])
            
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            sns.set(font='SimHei', style='white')
            fig = plt.figure(figsize=(12,7))
            ax1 = fig.add_subplot(111)
            ax1.patch.set_facecolor("gainsboro")
            for i in DfRoc.groupby('type'):
                ax1.plot(i[1]['FPR'], i[1]['recall'], label=i[0])
            ax1.plot([0,1], [0,1], 'k--')
            ax1.set_xlabel('False_Positive_Rate')
            ax1.set_ylabel('True_Positive_Rate')
            ax1.legend()
            ax1.set_title(title)
            plt.rcParams['savefig.facecolor'] = 'whitesmoke'
            plt.savefig(imagePath+'/ROC.png')
            plt.show()
        
        except Exception as e:
            print('H_ModelEvaluateData,ModelEffectClass,plot_roc',':',e) 


    def plot_pr_f1(self,dataInitial,flagName,factorName,dfName,imagePath):

        '''
        功能：绘制召回率、精确率、f1-score曲线图

        输入：
        dataInitial: 数据集 dataframe 
        flagName: 标签名称 string 
        factorName: 指标名称 string 
        dfName: 集合名称 string
        imagePath: 图片保存路径 string

        输出：
        召回率、精确率、f1-score曲线图，文件
        
        管理记录：
        1. created by hean 2021/07/07
        
        '''

        try:
        
            df = self.group_by_df(dataInitial, flagName, factorName)
            df['std']=list(map(lambda x,y,z:np.std([x,y,z])/float(np.mean([x,y,z])),df['recall'],df['precision'],df['F1_score']))
            stdmin=df['std'].min()
            minindex=df[df['std']==stdmin].index[0]
            bestcutoff=df[factorName][minindex]
            bestvalue=df['F1_score'][minindex]
            title = "PR-F1 Curve of %s\n bestvalue=%.2f , bestcutoff=%.2f" % (dfName, bestvalue, bestcutoff)
            
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            sns.set(font='SimHei', style='white')
            fig = plt.figure(figsize=(12,7))
            ax1 = fig.add_subplot(111)
            ax1.patch.set_facecolor("gainsboro")
            ax1.plot(df[factorName], df['recall'], label="Recall")
            ax1.plot(df[factorName], df['precision'], label="Precision")
            ax1.plot(df[factorName], df['F1_score'], label="F1-Score")
            ax1.set_xlabel('Score')
            ax1.set_ylabel('%Rate')
            ax1.legend()
            ax1.set_title(title)
            plt.rcParams['savefig.facecolor'] = 'whitesmoke'
            plt.savefig(imagePath+'/{}_PR_F1.png'.format(dfName))
            plt.show()
        
        except Exception as e:
            print('H_ModelEvaluateData,ModelEffectClass,plot_pr_f1',':',e) 


    def plot_ks_distribution(self, dataInitial, flagName, factorName, dfName, imagePath):

        '''
        功能：绘制累计好坏客户的曲线图，和分数分布曲线

        输入：
        dataInitial: 数据集 dataframe 
        flagName: 标签名称 string 
        factorName: 指标名称 string 
        dfName: 集合名称 string
        imagePath: 图片保存路径 string

        输出：
        累计好坏客户的曲线图，和分数分布曲线，文件
        
        管理记录：
        1. created by hean 2021/07/07
        
        '''

        try:
        
            resultDf = self.group_by_df(dataInitial, flagName, factorName)
            ks = max(resultDf['ks'])
            KsIndex=resultDf[resultDf['ks']==ks].index[0]
            KsValue=resultDf[factorName][KsIndex]
            
            title="Population Distribution of Good and Bad Accounts by Score Point of %s\nKS=%.2f , KS_cutoff=%s"%(dfName, ks, KsValue)
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            sns.set(font='SimHei', style='white')
            fig = plt.figure(figsize=(12,7))
            ax1 = fig.add_subplot(111)
            ax1.patch.set_facecolor("gainsboro")
            sns.kdeplot(dataInitial[factorName], shade=True,label='Score Distribution')
            ax1.set_xlabel('Score')
            ax1.set_ylabel('%Obs')
            ax1.legend(loc='upper left')
            ax1.set_title(title)
            ax2 = ax1.twinx()
            ax2.plot(resultDf[factorName], resultDf['cumsum_good'], color='g', label='Cumsum_Good')
            ax2.plot(resultDf[factorName], resultDf['cumsum_bad'], color='r', label='Cumsum_Bad')
            ax2.set_ylabel('%Cumulative Obs')
            ax2.legend(loc='upper right')
            plt.rcParams['savefig.facecolor'] = 'whitesmoke'
            plt.savefig(imagePath+'/%s_KS_Distribution.png'%dfName)
            plt.show()
        
        except Exception as e:
            print('H_ModelEvaluateData,ModelEffectClass,plot_ks',':',e) 


    def plot_good_bad_ind(self, dataInitial, flagName, factorName, dfName, imagePath):

        '''
        功能：绘制好客户、坏客户、中间客户分数分布图

        输入：
        dataInitial: 数据集 dataframe 
        flagName: 标签名称 string 
        factorName: 指标名称 string 
        dfName: 集合名称 string
        imagePath: 图片保存路径 string

        输出：
        好客户、坏客户、中间客户分数分布图，文件
        
        管理记录：
        1. created by hean 2021/07/07
        
        '''

        try:
        
            avggood = dataInitial[dataInitial[flagName]==0][factorName].mean()
            avgbad = dataInitial[dataInitial[flagName]==1][factorName].mean()
            title="Score Distribution of Accounts\nGood_Average=%.2f , Bad_Average=%.2f"%(avggood,avgbad)
            if len(dataInitial[dataInitial[flagName]==2])>0:
                avgind = dataInitial[dataInitial[flagName]==2][factorName].mean()
                title="Score Distribution of Accounts\n Good_Average=%.2f , Bad_Average=%.2f, Ind_Average=%.2f"%(avggood,avgbad,avgind)

            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            sns.set(font='SimHei', style='white')
            fig = plt.figure(figsize=(12,7))
            ax1 = fig.add_subplot(111)
            ax1.patch.set_facecolor("gainsboro")
            sns.kdeplot(dataInitial[dataInitial[flagName]==0][factorName], shade=True,color='green',label='Good Distribution')
            sns.kdeplot(dataInitial[dataInitial[flagName]==1][factorName], shade=True,color='red',label='Bad Distribution')
            if len(dataInitial[dataInitial[flagName]==2])>0:
                sns.kdeplot(dataInitial[dataInitial[flagName]==2][factorName], shade=True,color='blue',label='Ind Distribution')
            ax1.set_xlabel('Score')
            ax1.set_ylabel('%Obs')
            ax1.legend()
            ax1.set_title(title)
            plt.rcParams['savefig.facecolor'] = 'whitesmoke'
            plt.savefig(imagePath+'/%s_Distribution.png'%dfName)
            plt.show()
        
        except Exception as e:
            print('H_ModelEvaluateData,ModelEffectClass,plot_good_bad_ind',':',e) 


    def plot_cumulative_approve(self, dataInitial, flagName, factorName, dfName, imagePath):

        '''
        功能：绘制累计好坏通过率图

        输入：
        dataInitial: 数据集 dataframe 
        flagName: 标签名称 string 
        factorName: 指标名称 string 
        dfName: 集合名称 string
        imagePath: 图片保存路径 string

        输出：
        累计好坏通过率图，文件
        
        管理记录：
        1. created by hean 2021/07/07
        
        '''

        try:
        
            resultDf = self.group_by_df(dataInitial, flagName, factorName)
            resultDf['%Cumulative_Bad_Rate']=np.cumsum(resultDf['bad'])/np.cumsum(resultDf['bad']+resultDf['good'])
            resultDf['%Cumulative_Good_Rate']=np.cumsum(resultDf['good'])/np.cumsum(resultDf['bad']+resultDf['good'])
            title="Cumulative Good and Cumulative Bad Rate of Population"

            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            sns.set(font='SimHei', style='white')
            fig = plt.figure(figsize=(12,7))
            ax1 = fig.add_subplot(111)
            ax1.patch.set_facecolor("gainsboro")
            ax1.plot(resultDf[factorName],resultDf['%Cumulative_Bad_Rate'],color='red', lw=2,label="%Cumulative_Bad_Rate")
            ax1.plot(resultDf[factorName],resultDf['%Cumulative_Good_Rate'],color='green', lw=2,label="%Cumulative_Good_Rate")
            ax1.set_xlabel(u'Score')
            ax1.set_ylabel(u'%Cumulative_Rate')
            ax1.legend()
            ax1.set_title(title)
            plt.rcParams['savefig.facecolor'] = 'whitesmoke'
            plt.savefig(imagePath+'/%s_Cumulative_Approve.png'%dfName)
            plt.show()
        
        except Exception as e:
            print('H_ModelEvaluateData,ModelEffectClass,plot_cumulative_approve',':',e) 