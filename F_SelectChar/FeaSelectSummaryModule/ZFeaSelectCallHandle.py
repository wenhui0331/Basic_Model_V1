from F_SelectChar.FeaSelectProcessModule.FeaDisSelectHandle import FeaDisSelectClass
from F_SelectChar.FeaSelectProcessModule.FeaLinearSelectHandle import FeaLinearSelectClass
from F_SelectChar.FeaSelectProcessModule.FeaStabSelectHandle import FeaStabSelectClass
from F_SelectChar.FeaSelectProcessModule.FeaStepwiseSelectHandle import FeaStepwiseSelectClass
from F_SelectChar.FeaSelectProcessModule.FeaStepSelectWDHandle import WDFeaStepwiseSelectClass
from pandas.api.types import is_numeric_dtype
import re
import pandas as pd
import  numpy as np
from scipy.stats import stats
import statsmodels.api as sm    
import scipy
import os

class FeaSelectProcessClass(FeaDisSelectClass,FeaStabSelectClass,FeaLinearSelectClass,FeaStepwiseSelectClass):

    def __init__(self,ivThreshold=0.02,xgbSelectNum=None,psiThreshold=0.01,corrThreshold=0.6
					 ,vifThreshold=10,pThreshold=0.05,selection='stepwise',sle=0.05,sls=0.05,includes=[]):
        
            
        FeaDisSelectClass.__init__(self,ivThreshold,xgbSelectNum)
        FeaStabSelectClass.__init__(self,psiThreshold)
        FeaLinearSelectClass.__init__(self,corrThreshold,vifThreshold)
        FeaStepwiseSelectClass.__init__(self,pThreshold,selection,sle,sls,includes)
        
        
    """
    功能：
        将所有特征筛选的功能集和起来，进行特征筛选
        在此模块输入的数据为woe转换后的数据
        
        
    输入：
        ivThreshold：iv值筛选特征的区分度，默认参数为：0.02
        psiThreshold：psi值筛选特征的稳定性，默认参数为：0.1
        corr_Threshold：特征的相关性筛选，保留IV值高的特征，默认参数为0.6
        vifThreshold: 特征VIF筛选的阈值，默认为10
        xgbSelectNum:xgb模型特征重要性特征筛选数量，默认为None
        pThreshold: 显著性检验参数，默认0.05
        selection：逐步回归的方式，默认为逐步回归
        sle：特征进入模型的标准即统计意义水平值，默认0.05
        sls：特征在模型中保留的标准即统计意义水平值，默认0.05
        includes：逐步回归时强制进入模型的特征 
        
    输出：
    feature_select：输出通过所有特征筛选的过程

    管理记录：
    1. edited by 唐杰君 2021/06/23

    """
    
        
    def feature_select(self,dataForSelect,woeTest,target,drop_reason=True,return_cal_value=True):    

        """
        功能： 
            特征筛选
        输入：
            xdataForSelect：特征筛选的数据集，训练集
            woeTest：测试集数据
            target：目标特征
            drop_reason：是否需要保留删除原因，默认为True
            return_cal_value：是否需要保留各项指标，默认为True
        输出：
            dataForSelect: 保留数据集
            feat_nocorr：经过一系列特征筛选，全部留下的变量
            psiDropVar： 经过PSI剔除的变量
            corrDropVar：经过相关系数剔除的变量
            vifDropVar：经过膨胀系数剔除的变量
            pDropVar：经过置信度剔除的变量
            stepwiseDropVar：经过逐步回归剔除的变量
            coefDropVar：经过符号不一致剔除的变量
            resDataAll：保留的数据，特征删除原因以及各项指标值

        管理记录：
        1. edited by 唐杰君 2021/06/23
        2. modify by 王文丹 2021/07/25  增加输出
        3. modify by 王文丹 2021/09/13  替换置信度和逐步回归筛选的函数
        
        """

        try:
            ivDropVar = psiDropVar = corrDropVar = vifDropVar = stepwiseDropVar = pDropVar =xgbDropVar \
            = []
            
            ivlist =pd.DataFrame({'variable':[],'iv_value':[]})
            psilist =pd.DataFrame({'variable':[],'psi_value':[]})
            xgb_fea_sort =pd.DataFrame({'variable':[],'fea_importance':[]})
            calvalue = pd.DataFrame({'variable':list(dataForSelect.drop(columns=[target]).columns)})

            print('数据打印0',dataForSelect.shape)
            
            print('___________________________1.区分度筛选特征（iv特征筛选）_____________________________')
            if self.ivThreshold is not None:
                ivDropVar,dataForSelect,ivlist = self.iv_select(dataForSelect,target)
            print('iv删除特征:' ,ivDropVar)

            print('数据打印1',dataForSelect.shape)

            print('___________________________2.区分度筛选特征（xgb特征筛选）_____________________________')
            if self.xgbSelectNum is not None:
                xgbDropVar,dataForSelect,xgb_fea_sort =self.xgb_select(dataForSelect,woeTest,target)
            print('xgb删除特征:' ,xgbDropVar)

            print('数据打印2',dataForSelect.shape)
              
            print('___________________________3.稳定性筛选特征（psi特征筛选）_____________________________')
            if self.psiThreshold is not None:
                psiDropVar,dataForSelect,psilist =self.psi_select(dataForSelect,woeTest,target)
            print('psi删除特征:' ,psiDropVar)

            print('数据打印3',dataForSelect.shape)
            
            print('___________________________4.线性关系筛选特征（相关系数特征筛选）_____________________________') 
            if self.corrThreshold is not None:
                corrDropVar,dataForSelect,corrDesc = self.corr_select(dataForSelect,target,by='iv')
            print('相关系数删除特征:' ,corrDropVar)

            print('数据打印4',dataForSelect.shape)
            
            print('___________________________5.线性关系筛选特征（vif特征筛选）_____________________________')
            if self.vifThreshold is not None:
                vifDropVar,dataForSelect,vifList = self.vif_select(dataForSelect,target)
            print('vif删除特征:' ,vifDropVar)  

            print('数据打印5',dataForSelect.shape)  
           
            # print('___________________________6.逐步回归筛选特征_____________________________________________')
            # if self.selection is not None:
            #     stepwiseDropVar ,pDropVar , dataForSelect = self.stepwise_select(dataForSelect,target)
            # print('stepwise删除特征:' ,stepwiseDropVar) 
            
            print('___________________________6.置信度+逐步回归筛选特征_____________________________________________')
            if self.selection is not None:
                FSSC = WDFeaStepwiseSelectClass(self.pThreshold,target)
                colForStepwiseValue = list(dataForSelect.columns)  #上述筛选留下的数据集
                colForStepwiseValue.remove(target) #去掉标签名称
                (dataForSelect,feat_nocorr,pDropVar,stepwiseDropVar,coefDropVar) = FSSC.AllLinearStepSelectClass(dataForSelect,colForStepwiseValue)

            resDataAll = (dataForSelect,)    
            
            
            if drop_reason:
                dropreason = {
                    'iv删除特征': ivDropVar,
                    'xgb删除特征':xgbDropVar,
                    'psi删除特征':psiDropVar,
                    '相关系数删除特征': corrDropVar,
                    'vif删除特征':vifDropVar,
                    '逐步回归删除特征':stepwiseDropVar,
                    '显著性检验删除特征':pDropVar,
                    '符号不一致':coefDropVar
                }
                resDataAll+= (dropreason,)
                
            if return_cal_value:
                
                
                calvalue = calvalue.merge(ivlist,on='variable',how='left')\
                                 .merge(xgb_fea_sort ,on ='variable',how='left')\
                                 .merge(psilist,on='variable',how='left')\
                                 .merge(vifList,on='variable',how='left')
                                 
                calvalue = calvalue.sort_values(by=['iv_value','fea_importance'],ascending=False)
                resDataAll +=(calvalue,)
                
            # #保存到excel
            # dropeasondf = pd.DataFrame()
            # for i in resDataAll[1].keys():
            #     df1 = pd.DataFrame( resDataAll[1][i],columns=[i])
            #     dropeasondf = pd.concat([dropeasondf,df1],axis=1)
            # writer = pd.ExcelWriter('C:/Users/Administrator/Desktop/特征筛选汇总.xlsx')
            # resDataAll[2].to_excel(writer,sheet_name='特征指标值')
            # dropeasondf.to_excel(writer,sheet_name='特征筛选过程')
            # writer.save()
            # writer.close()
            return (dataForSelect,feat_nocorr,psiDropVar,corrDropVar,vifDropVar,pDropVar,stepwiseDropVar,coefDropVar,resDataAll)
            
            
        except Exception as e:
            print('F_SelectChar,FeaSelectSummaryModule,ZFeaSelectCallHandle,FeaSelectProcessClass,feature_select',':',e)       
		