import pandas as pd
import  numpy as np
from scipy.stats import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from F_SelectChar.FeaSelectProcessModule.FeaDisSelectHandle import FeaDisSelectClass


class FeaLinearSelectClass():
    
    """
    功能：
        特征筛选：通过相关系数和vif进行特征筛选
        在此模块输入的数据为woe转换后的数据
        
    输入：
        corrThreshold：相关系数筛选阈值,默认为0.6
        vifThreshold：vif筛选阈值，默认为10
        
    输出：
    函数corr_select：相关系数筛选特征，保留Iv值高的特征
    函数vif_select： 筛选vif值不符合条件的特征
        
    维护记录：
    1. modify by 唐 2021/06/23    
    """
    
    def __init__(self,corrThreshold=0.6,vifThreshold = 10):
    
        self.corrThreshold = corrThreshold
        self.vifThreshold = vifThreshold
        


    def corr_select(self,dataForSelect,target,by='iv'):
    
        """
        功能：
            相关系数筛选，若相关系数高于阈值，选取iv高的特征
            
        输入:
            dataForSelect:相关系数筛选的数据集
            target:目标特征
            
        输出：
            corrDropVar：通过相关系数删除的特征
            corrData:相关系数筛选过后保留的数据集
            corrDesc:保留后数据的相关系数
        管理记录：
        1. edited by 唐杰君 2021/06/23
        """
        try:
            corrThreshold = self.corrThreshold
            # print(dataForSelect)
            
            if by.upper()=='IV':
                ivlist = FeaDisSelectClass.iv_claculate(dataForSelect, target = target)
                series_ = ivlist['variable']
            
            drop_set,var_set=set(),set(series_)
            
            for i in series_:
                if i in var_set:
                    var_set.remove(i)
                    
                if i not in drop_set:
                    drop_set |={v for v in var_set if stats.pearsonr(dataForSelect[i].values,dataForSelect[v].values)[0] >= corrThreshold}
                    var_set -=drop_set
            #print('\n相关性剔除特征：\n', drop_set)
            corrDropVar = list(drop_set)
            corrlist =  [i for i in series_ if i not in drop_set]
            corrData = dataForSelect[corrlist+[target]]
            corrDesc = corrData.drop(columns=[target]).corr()

            return (corrDropVar,corrData,corrDesc)
            
        except Exception as e:
            print('F_SelectChar,FeaSelectProcessModule,FeaLinearSelectHandle,FeaLinearSelectClass,corr_select',':',e)  
            
            
    
    def vif_select(self,dataForSelect,target):
        
        """
        功能：
            通过vif值筛选特征，并输出筛选后的数据集
            
        输入：
            dataForSelect：需要通过vif筛选的训练数据集（woe转换后的数据集）
            target： 目标特征
            
        输出：
        vifDropVar：通过vif删除的特征
        vifData：通过vif筛选后保留的数据集
        vifList: vif值

        管理记录：
        1. edited by 唐杰君 2021/06/23
        """        
    
        try:
                           
            x_data = dataForSelect.drop(columns=[target])
            x_data['c'] = 1
            
            name = x_data.columns
            #print(x_data.shape)
            x = np.matrix(x_data)
            VIF_list = [variance_inflation_factor(x,i) for i in range(x.shape[1])]
            vifList = pd.DataFrame({'variable':name,"vifList":VIF_list})
            vifList = vifList[vifList['variable']!='c']
            vifDropVar = list(vifList[vifList['vifList'] >= self.vifThreshold].variable)
            vifData = dataForSelect[list(set(dataForSelect.columns)-set(vifDropVar))]

            return vifDropVar ,vifData,vifList
            
        except Exception as e:
            print('F_SelectChar,FeaSelectProcessModule,FeaLinearSelectHandle,FeaLinearSelectClass,vif_select',':',e)        

            