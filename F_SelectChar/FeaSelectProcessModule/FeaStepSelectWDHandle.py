import pandas as pd
import  numpy as np
from scipy.stats import stats
from statsmodels.api import Logit
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm    

class WDFeaStepwiseSelectClass():

    """
    功能：特征筛选：线性回归+显著性筛选+线性回归+向前逐步筛选

    输入：

    输出：

    管理记录：
    1. edited by 王文丹 2021/09/13
    
    """

    def __init__(self,pThreshold,ylabel):

        self.pThreshold = pThreshold
        self.ylabel = ylabel

    def AllLinearStepSelectClass(self,data,feat_weakcorr_ivsorted1):
        """
        功能：线性回归+显著性筛选+线性回归+向前逐步筛选 总调用函数

        输入：

        输出：

        管理记录：
        1. edited by 王文丹 2021/09/13
        
        """
        # 线性回归+逐步筛选+线性回归
        (feat_nocorr,pDropVar,summary) = self.linear_regression_significance(data,feat_weakcorr_ivsorted1)

        # 向前逐步回归筛选
        (feat_nocorr,StepWiseDropVar)=self.forward_delete_pvalue(data[feat_nocorr],data[self.ylabel])

        # 逻辑回归系数符号筛选,在筛选前需要做woe转换
        (feat_nocorr,coefDropVar,lr_coe) = self.forward_delete_coef(data[feat_nocorr],data[self.ylabel])
        print(lr_coe)

        dataForSelect = data[feat_nocorr+[self.ylabel]]

        return (dataForSelect,feat_nocorr,pDropVar,StepWiseDropVar,coefDropVar)

    def linear_regression_significance(self,data,feat_weakcorr_ivsorted1):

        '''
        功能：主要是显著性筛选

        输入：data 训练集合
        输入：feat_weakcorr_ivsorted1 经过相关性筛选、多重线性筛选后留下的特征名称WOE
        输入：ylabel 好坏标签特征名称
        输入：pThreshold 置信度 默认0.05
        
        输出：feat_nocorr 留下经过相关性筛选、多重共线性筛选、显著性筛选后变量
        输出：feat_large_p 变量p值字典

        管理记录：
        1. edited by 王文丹 2021/09/13
        
        '''
        #feat_weakcorr_ivsorted1至此已排除具有明显多重共线性
        feat_nocorr=feat_weakcorr_ivsorted1.copy()
        x=data[feat_nocorr]
        x['intercept']=[1]*x.shape[0]
        y=data[self.ylabel]
        lr=Logit(y,x).fit()
        print('##################################################################')
        print('第一次线性回归的汇总结果：')
        print(lr.summary())

        print('##################################################################')
        #剔除不显著特征，取显著性水平pThreshold
        feat_large_p={k:v for k,v in lr.pvalues.to_dict().items() if v > self.pThreshold}
        feat_large_p=sorted(feat_large_p.items(),key=lambda x:x[1],reverse=True)
        # 存储显著性pvalue筛选后的变量
        pDropVar = []
        while len(feat_large_p)>0:
            feat_max_p=feat_large_p[0][0]
            print('不显著变量：%s，p>%s'%(feat_max_p,self.pThreshold))
            if feat_max_p=='intercept':
                print('intercept 不显著')
            else:     
                feat_nocorr.remove(feat_max_p)  # 显著性筛选留下的变量
                pDropVar.append(feat_max_p)     # 显著性筛选剔除的变量
                x_p=data[feat_nocorr]
                x_p['intercept']=[1]*x_p.shape[0]
                lr_p=Logit(y,x_p).fit()
                feat_large_p={k:v for k,v in lr_p.pvalues.to_dict().items() if v>self.pThreshold}
                feat_large_p=sorted(feat_large_p.items(),key=lambda x:x[1],reverse=True)
                
        x_smallP=data[feat_nocorr]
        x_smallP['intercept']=[1]*x_smallP.shape[0]
        lr_smallP=Logit(y,x_smallP).fit()
        print('##################################################################')
        print('经过显著性筛选后的线性回归汇总结果：')
        #所有特征p值都小于self.pThreshold，表示所有特征都显著；回归系数除截距外符号都ok
        print(lr_smallP.summary())
        
        return (feat_nocorr,pDropVar,lr_smallP.summary())
    
    # 向前逐步回归筛选
    def forward_delete_pvalue(self,x_train,y_train):
        """
        x_train -- x训练集
        y_train -- y训练集
        
        return :显著性筛选后的变量

        管理记录：
        1. edited by 王文丹 2021/09/13

        """
        col_list = list(x_train.columns)
        pvalues_col=[]
        StepWiseDropVar = []
        for col in col_list:
            pvalues_col.append(col)
            x_train2 = sm.add_constant(x_train.loc[:,pvalues_col])
            sm_lr = sm.Logit(y_train,x_train2)
            sm_lr = sm_lr.fit()
            for i,j in zip(sm_lr.pvalues.index[1:],sm_lr.pvalues.values[1:]): 
                if j>=0.05:
                    pvalues_col.remove(i)      #经过逐步回归留下的变量
                    StepWiseDropVar.append(i)  #经过逐步回归删除的变量
        
        x_new_train = x_train.loc[:,pvalues_col]
        x_new_train2 = sm.add_constant(x_new_train)
        lr = sm.Logit(y_train,x_new_train2)
        lr = lr.fit()
        print(lr.summary2())
        return (pvalues_col,StepWiseDropVar)
    
    # 逻辑回归系数符号筛选,在筛选前需要做woe转换
    def forward_delete_coef(self,x_train,y_train):
        """
        x_train -- x训练集
        y_train -- y训练集
        
        return :
        coef_col: 回归系数符号筛选后的变量
        coefDropVar: 回归系数符号剔除后的变量 
        lr_coe：每个变量的系数值

        管理记录：
        1. edited by 王文丹 2021/09/13

        """
        col_list = list(x_train.columns)
        coef_col = []
        coefDropVar = []
        for col in col_list:
            coef_col.append(col)
            x_train2 = x_train.loc[:,coef_col]
            sk_lr = LogisticRegression(random_state=0).fit(x_train2,y_train)
            coef_df = pd.DataFrame({'col':coef_col,'coef':sk_lr.coef_[0]})
            if coef_df[coef_df.coef>0].shape[0]>0:
                coef_col.remove(col)
                coefDropVar.append(col)
                
        x_new_train = x_train.loc[:,coef_col]
        lr = LogisticRegression(random_state=0).fit(x_new_train,y_train)
        lr_coe = pd.DataFrame({'col':coef_col,
                            'coef':lr.coef_[0]})
        return (coef_col,coefDropVar,lr_coe)







