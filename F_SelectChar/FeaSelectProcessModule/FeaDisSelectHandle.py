import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

class FeaDisSelectClass():
    
    """
    功能描述：特征筛选：包括IV值、xgboost模型变量筛选
             在此模块输入的数据为woe转换后的数据
        
    输入：
    ivThreshold：iv值筛选变量的区分度，默认参数为：0.02
    xgbSelectNum:xgb模型特征重要性特征筛选数量，默认为None   
        
    输出：
    函数iv_xy：计算x变量和y变量的iv值
    函数iv_claculate： 数据集中每个变量的iv值
    函数iv_select ：通过iv计算，剔除低iv值变量以及剔除后的数据集
    函数xgb_select：xgb模型的特征重要性筛选变量
        
    管理记录：
    1. modify by 唐杰君 2021/06/23    
    """
    def __init__(self,ivThreshold =0.02,xgbSelectNum=None):
        
        
        self.ivThreshold = ivThreshold
        self.xgbSelectNum = xgbSelectNum
    
    
    @staticmethod    
    def iv_xy(x, ylabel):
        """
        功能描述：计算变量的IV值，只能单个特征计算（此处的数据是分箱好的数据，如col_Bin 或 col_Bin_WOE）

        输入：
        x：计算iv的变量 pandas  格式如df['c1']  
        ylabel：目标变量 pandas  格式如df['y']

        输出：
        iv_total：iv值 float
        
        管理记录：
        1. edited by 唐杰君 2021/06/23
        2. modify by 王文丹 2021/07/25 修正变量名称、补全注释
        """
        try:
            # good bad func
            def goodbad(df):
                names = {'good': (df['y']==0).sum(),'bad': (df['y']==1).sum()}
                return pd.Series(names)
            
            # iv 计算
            ivTotal = pd.DataFrame({'x':x.astype('str'),'y':ylabel}).fillna('missing') \
            .groupby('x').apply(goodbad).replace(0, 0.9).assign(
                DistrBad = lambda x: x.bad/sum(x.bad),
                DistrGood = lambda x: x.good/sum(x.good)
            ) \
            .assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood)) \
            .iv.sum()
            # return iv
            return ivTotal
        except Exception as e:
            print('F_SelectChar, FeaSelectProcessModule, FeaDisSelectHandle, F_SelectChar, FeaDisSelectClass, iv_xy',':',e)            
   
    
    @staticmethod 
    def iv_claculate(dataForSelect, target, X=None, order=True):
    
        """
        功能描述：计算输入的数据集中每个变量的iv值

        输入：
        dataForSelect：需要计算Iv值的Woe  DataFrame
        target:目标变量 str格式如'y'
        X:需要计算iv值的变量，默认值为NOne,则为全部变量
        order：输出值是否按iv值排序，默认为True

        输出:
        ivList:dt中各变量的iv值 DataFrame ['variable','iv_value']  
        
        管理记录：
        1. modify by 唐杰君 2021/06/23   
        2. edited by 王文丹 2021/07/24 将target修正为string 
        """
        try:
            dataForSelect = dataForSelect.copy(deep=True)
            if isinstance(target,str):
                target = [target]
            if isinstance(X, str) and X is not None:
                X = [X]
            if X is not None: 
                dataForSelect = dataForSelect[target+X]
              
            xs = list(set(dataForSelect.columns) - set(target))
            # info_value
            ivList = pd.DataFrame({
                                    'variable': xs,
                                    'iv_value': [FeaDisSelectClass.iv_xy(dataForSelect[i], dataForSelect[target[0]]) for i in xs]
                                    }, columns=['variable', 'iv_value'])
            # sorting iv
            if order: 
                ivList = ivList.sort_values(by='iv_value', ascending=False)
            return ivList 
        except Exception as e:
            print('F_SelectChar, FeaSelectProcessModule, FeaDisSelectHandle, F_SelectChar, FeaDisSelectClass, iv_claculate',':',e)           

    def iv_select(self,dataForSelect,target):
        
        """
        功能描述：通过iv值筛选变量，并输出筛选后的数据集

        输入：
        dataForSelect：需要通过iv筛选的数据集（woe转换后的数据集）
        target： 目标变量

        输出：
        ivDropVar：通过iv删除的变量
        IvWoeData：通过iv筛选后保留的数据集
        ivList：每个变量的iv值

        管理记录：
        1. modify by 唐杰君 2021/06/23
        """
    
        try:
            ivThreshold = self.ivThreshold
            ivList = FeaDisSelectClass.iv_claculate(dataForSelect, target = target)    
            ivDropVar = list(ivList[ivList['iv_value']<=ivThreshold]['variable'] )  
            ivhold = list(set(ivList['variable'])-set(ivDropVar))
            IvWoeData = dataForSelect[ivhold+[target]]
            return ivDropVar,IvWoeData,ivList
            
        except Exception as e:
            print('F_SelectChar, FeaSelectProcessModule, FeaDisSelectHandle, F_SelectChar, FeaDisSelectClass, iv_select',':',e)         
        
        
    def xgb_select(self,dataForSelect,woeTest,target):
    
        """
        功能：
            通过xgb模型的特征重要性筛选变量
        输入：
            dataForSelect：需要通过Xgb筛选的数据集（woe转换后的数据集）
            woeTest：数据测试集
            target： 目标变量   
        输出：
            xgbDropVar：xgb删除的特征
            xgbWoeData：通过xgb模型特征重要性删选后的数据
            xgbFeaSort：xgb特征重要性
        """
        try :           
            params={
                    'max_depth':5,
                    'lambda':10,
                    'subsample':0.5,
                    'colsumple_bytree':0.3,
                    'min_child_weight':20,
                    'eta':0.025,
                    'seed':0,
                    'nthread':8}
                    
            model=XGBClassifier(params=params)
            
            X_train = dataForSelect.drop(columns= [target])
            y_train = dataForSelect[target]
            var_info=X_train.columns.tolist()
            X_test = woeTest[var_info]
            y_test = woeTest[target]    
            eval_set=[(X_train,y_train),(X_test,y_test)]
            model.fit(X_train,y_train,eval_metric=['auc'],eval_set=eval_set,verbose=True)
            
            var_info=X_train.columns.tolist()
            fea_imp_list = model.feature_importances_.tolist()
            
            info_fea=pd.DataFrame({'variable':var_info,'fea_importance':fea_imp_list})
        
            xgbFeaSort=info_fea.sort_values(by=['fea_importance'],ascending=False) 
            
            xgbselectvar = list(xgbFeaSort.iloc[:self.xgbSelectNum]['variable'])
            xgbDropVar = list(set(var_info)-set(xgbselectvar))
            xgbWoeData= dataForSelect[xgbselectvar+[target]]
            xgbFeaSort.to_excel('ouput/F_output/xgb特征重要性.xlsx')
            return xgbDropVar ,xgbWoeData,xgbFeaSort
            
        except Exception as e:
            print('F_SelectChar,FeaDisSelectClass,xgb_select',':',e)           
        
        