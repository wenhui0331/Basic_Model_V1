from pandas.api.types import is_numeric_dtype
import re
import pandas as pd
import  numpy as np
from scipy.stats import stats
import statsmodels.api as sm    
import scipy
import os


class FeaStepwiseSelectClass():

    """
    功能：
        特征筛选：通过stepwise进行特征筛选
        在此模块输入的数据为woe转换后的数据
        
    输入：
        pThreshold: 显著性检验参数，默认0.05
        selection：逐步回归的方式，默认为逐步回归
        sle：变量进入模型的标准即统计意义水平值，默认0.05
        sls：变量在模型中保留的标准即统计意义水平值，默认0.05
        includes：逐步回归时强制进入模型的变量   
        
    输出：
    函数stepwise：stepwise输出结果
    函数stepwise_select： stepwise筛选出的特征，p值检验通过的特征
    维护记录：
    1. modify by 唐 2021/06/23    
    """

    def __init__(self,pThreshold=0.05,selection='stepwise',sle=0.05,sls=0.05,includes=[]):
    
        self.pThreshold = pThreshold
        self.selection = selection
        self.sle = sle
        self.sls = sls
        self.includes = includes
    
    @staticmethod 
    def score_test(xData,yData,yPred,DF=1):
    
        """
        功能：
            计算自由度为df的卡方分布在任意点处的分布函数值
        输入：
            xData：测试集的X变量集
            yData：测试集的目标变量
            yPred：预测变量
            DF；自由度
        输出：
            chi2_0[0,0]：卡方值
            DF：自由度
            Pvalue：自由度为1的卡方分布在 chi2_0[0,0]点处的分布函数值
        """
        try:
            x_arr=np.matrix(xData)
            y_arr=np.matrix(yData).reshape(-1,1)
            yh_arr=np.matrix(yPred).reshape(-1,1)
            grad_0=-x_arr.T * (y_arr-yh_arr)
            info_0=np.multiply(x_arr,np.multiply(yh_arr,(1-yh_arr))).T * x_arr
            cov_m=info_0**(-1)
            chi2_0=grad_0.T * cov_m * grad_0
            Pvalue=(1-scipy.stats.chi2.cdf(chi2_0[0,0],DF))
            return(chi2_0[0,0],DF,Pvalue)
        
        except Exception as e:
            print('F_SelectChar,FeaStepwiseSelectClass,score_test',':',e)   


            
    def stepwise(self,xData, yData):
    
        """
        功能：
            逐步回归
        输入：
            xData：测试集的X变量集
            yData：测试集的目标变量

        输出：
            res:各项指标的统计值
        """
        try:
            selection = self.selection
            sle = self.sle
            sls =self.sls
            includes = self.includes
        
        ###检查x和y的长度
            if len(xData) != len(yData):
                print('x,y长度不一致!')
            xData = xData.copy()
            yData = yData.copy()
            ###检查x
            if isinstance(xData,pd.core.frame.DataFrame) == True:
                x_list = list(xData.columns.copy())
            elif isinstance(xData,np.ndarray) == True:
                if len(xData.shape) == 1:
                    x_list = ['x_0']
                    xData = pd.DataFrame(xData.reshape(-1,1),columns=x_list)
                elif len(xData.shape) == 2:
                    x_list = ['x_' + str(i) for i in np.arange(xData.shape[1])]
                    xData = pd.DataFrame(xData,columns=x_list)
                else:
                   print('x有问题!')
            else:
                print('x有问题!')
            ###处理强制进入变量
            try:
                if (includes>0) and (includes>0) :
                    includes = x_list[:includes].copy()
                else:
                    includes = []
            except:
                pass
            ###处理x,y
            xData['_const']=1
            if (isinstance(yData,pd.core.frame.DataFrame) == True) or (isinstance(yData,pd.core.series.Series) == True):
                yData = yData.values.reshape(-1,1)
            else:
                yData = yData.reshape(-1,1)
            ####stepwise
            if selection.upper() == 'STEPWISE':
                include_list = ['_const'] + includes
                current_list = []
                candidate_list = [_x for _x in x_list if _x not in include_list]
                ##第一次拟合
                lgt = sm.Logit(yData,xData[include_list])
                print(xData[include_list].shape)
                res = lgt.fit()
                ##预测结果
                yPred = res.predict()
                ####输出第一步的拟合结果以及卡方检验结果
                print('========================第0步结果=======================')
                print(res.summary2())
                print(res.wald_test_terms())
                ##循环增删变量
                STOP_FLAG = 0 
                step_i = 1
                while(STOP_FLAG == 0):
                    if len(candidate_list) == 0:
                        break
                    ##遍历所有候选变量，计算每一个加入的候选变量对应的score统计量
                    score_list = [FeaStepwiseSelectClass.score_test(xData[include_list + current_list +[x0]]
                                            ,yData
                                            ,yPred)
                                    for x0 in candidate_list]
                    score_df = pd.DataFrame(score_list,columns=['chi2','df','p-vlue'])
                    score_df['xvar'] = candidate_list
                    slt_idx = score_df['chi2'].idxmax()
                    p_value = score_df['p-vlue'].iloc[slt_idx]
                    enter_x = candidate_list[slt_idx]
                    ###
                    ####输出第i步的候选变量的score统计量
                    print('#####======第' + str(step_i) + '步：候选变量的遍历结果=======')
                    print(score_df[['xvar','chi2','df','p-vlue']])
                    if p_value <= sle:
                        current_list.append(enter_x)   ##加入模型列表
                        candidate_list.remove(enter_x) ##从候选变量列表平中删除
                        print('######====第' + str(step_i) + '步：加入变量' + enter_x)
                    else:
                        STOP_FLAG = 1
                        print('######====第' + str(step_i) + '步：未加入变量')
                    ##根据新的变量列表，重新拟合，查看是否所有变量都能保留
                    lgt = sm.Logit(yData,xData[include_list+current_list])
                    res = lgt.fit()
                    ##预测结果
                    yPred = res.predict()
                    ##wald chi2 test
                    chi2_df = res.wald_test_terms().table.copy()
                    ##检查是否有变量需要删除
                    tmp_del_list = [tmp_x for tmp_x in chi2_df.index if tmp_x not in include_list]
                    ####输出第i步的候选变量的score统计量
                    print('######======第' + str(step_i) + '步：wald卡方检验======')
                    #print(res.wald_test_terms())
                    ##如果p-value大于等于sls,删除最大的
                    if len(tmp_del_list) > 0:
                        tmp_chi2 = chi2_df.loc[tmp_del_list].sort_values(by='statistic')
                        if tmp_chi2['pvalue'].iloc[0] >sls:
                            del_x = tmp_chi2.index[0]
                            ##打印结果
                            print('######====第' + str(step_i) + '步：删除变量' + del_x)
                            ##如果删除的是最近加入的变量，则停止筛选
                            if del_x == current_list[-1]:
                                current_list.remove(del_x)
                                STOP_FLAG = 1
                            else:
                                current_list.remove(del_x)
                            ##删除的变量加入候选变量列表中
                            candidate_list.append(del_x)
                            
                            ###根据删除后的变量列表再次拟合
                            lgt = sm.Logit(yData,xData[include_list+current_list])
                            res = lgt.fit()
                            ##预测结果
                            yPred = res.predict()
                        else:
                            print('######====第' + str(step_i) + '步：未删除变量' )
                    else:
                        print('######====第' + str(step_i) + '步：未删除变量' )
                    print('########################################')
                    print()
                    step_i += 1
                print('========================最终结果汇总=======================')
                print(res.summary2())
                print(res.wald_test_terms())
            ####简单逻辑回归
            else:
                lgt = sm.Logit(yData,xData)
                res = lgt.fit()
                print('========================最终结果汇总=======================')
                print(res.summary2())
                print(res.wald_test_terms())
            return res
            
        except Exception as e:
            print('F_SelectChar,FeaStepwiseSelectClass,stepwise',':',e)    			
			
    def stepwise_select(self,dataForSelect,target):
    
        """
        功能：
            通过stepwise进行变量选择
        输入:
            dataForSelect: 变量筛选的数据集
            target:目标变量
        输出：
            stepwiseDropVar:通过stepwise删除的变量
            pDropVar:p值显著性检验删除的变量
            stepwiseData: 通过stepwise保留的数据集 
        管理记录：
            1. edited by 唐杰君 2021/06/23       
        """
        try:
            xData = dataForSelect.drop(columns=[target])
            yData = dataForSelect[target]
            print(xData.shape)
            stepwise_res = self.stepwise(xData, yData)
            stepwise_var = list(stepwise_res.params[stepwise_res.params>0].index)
            stepwiseDropVar = list(set(xData.columns)-set(stepwise_var))
            pDropVar = list(stepwise_res.pvalues[stepwise_res.pvalues>self.pThreshold].index)
            stepwiseData = dataForSelect[list(set(stepwise_var)-set(pDropVar))+[target]]
            return stepwiseDropVar,pDropVar ,stepwiseData
            
        except Exception as e:  
            print('F_SelectChar,FeaSelectProcessModule,FeaStepwiseSelectHandle,FeaStepwiseSelectClass,stepwise_select',':',e) 			

