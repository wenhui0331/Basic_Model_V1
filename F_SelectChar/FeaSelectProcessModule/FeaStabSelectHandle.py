
import pandas as pd
import numpy as np

class FeaStabSelectClass():

    """
    功能：
        特征筛选：特征稳定性筛选,Psi筛选变量
        在此模块输入的数据为woe转换后的数据
        
    输入：
        psiThreshold：psi值筛选变量的区分度，默认参数为：0.02
        
    输出：
    函数psi_claculate：计算每个变量的psi值
    函数psi_select：psi值筛选变量，并输出筛选后的数据集
    
    """
	 
    
    def __init__(self,psiThreshold =0.1):
        
        self.psiThreshold = psiThreshold

    @staticmethod 
    def psi_claculate(woeBaseData, woeTestData,target):
    
        """
        功能:
            计算每个变量的psi值
        输入:
            woeBaseData:基准数据集
            woeTestData:比较数据集
            target:目标变量
        输出：
            psiList:每个变量的psi

        管理记录：
        1. edited by 唐杰君 2021/06/23
        2. modify by 王文丹 2021/07/25 修正易造成歧义的变量命名以及加上错误抛出      
        """
        try:
        
            psivalue=list()
            psivar = woeBaseData.columns.drop(target)
            
            for col in psivar:
                base_prop = pd.Series(woeBaseData[col]).value_counts(normalize = True, dropna = False)
                test_prop = pd.Series(woeTestData[col]).value_counts(normalize = True, dropna = False)
        
                psi = np.sum((test_prop - base_prop) * np.log(test_prop / base_prop))
        
                frame = pd.DataFrame({
                    'baseData': base_prop,
                    'testData': test_prop,
                    })
                frame.index.name = 'value'
                
                psivalue.append(psi)
            psiList = pd.DataFrame({'variable': psivar
                            ,'psi_value':psivalue
                            },columns=['variable', 'psi_value'])
            return psiList
        
        except Exception as e:

            print('F_SelectChar,FeaSelectProcessModule,FeaStabSelectHandle,FeaStabSelectClass,psi_claculate',':',e)   
		
		
    def psi_select(self,woeTrainData,woeTestData,target):
    
        """
        功能：
            通过psi值筛选变量，并输出筛选后的数据集
        输入：
            woeTrainData：需要通过psi筛选的训练数据集（woe转换后的数据集）
            testTestData: 测试集数据
            target： 目标变量
        输出：
        psiDropVar：通过psi删除的变量
        psiData：通过psi筛选后保留的数据集
        psiList：每个变量的psi值

        管理记录：
        1. edited by 唐杰君 2021/06/23
        2. modify by 王文丹 2021/07/25 修正易造成歧义的变量命名
        """
        
        try:
        
            psiThreshold = self.psiThreshold
            psiList = FeaStabSelectClass.psi_claculate(woeTrainData,woeTestData, target = target)    
            psiDropVar = list(psiList[psiList['psi_value']>psiThreshold]['variable'] )  
            psihold = list(set(psiList['variable'])-set(psiDropVar))
            psiData = woeTrainData[psihold+[target]]
            return psiDropVar,psiData,psiList
        
        except Exception as e:
            print('F_SelectChar,FeaSelectProcessModule,FeaStabSelectHandle,FeaStabSelectClass,psi_select',':',e)    