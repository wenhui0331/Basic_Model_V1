import pandas as pd
import numpy as np


class SplitPointApplyClass():

    """
    功能描述：运用分箱切点，将数据映射为对应的分箱点

    输入：
    charValue: 数值 （某个特征下的具体数值）
    cutOffPoints: 分割点
    specialValue：特殊值，如缺失值
    ylabel: y标签的特征名称 char
    colForSplit: 用于分箱的特征名称 char
    colBinForSplit: 用于分箱的特征名称+'_Bin' char

    输出：
    函数split_assign：将数据映射为对应的分箱点
    函数split_Bin: 将分箱的切分点运用至数据集上，形成splitCharname_Bin分箱特征
    函数split_index：根据数据大小，形成分箱结果groupby的索引,使得分箱结果展示按照正确的大小排序
    函数split_result：分箱切点对应区间的好坏计算
    函数split_result：分箱结果汇总计算/IV值/WOE值的映射字典
    函数combine_zero：对存在单箱的坏样本数（好样本数）的切点进行合并
    函数monotone_check：检查是否分箱结果是否单调

    管理记录：
    1. edited by 王文丹 2021/07/08
    
    """
    def __init__(self, specialValue, ylabel):

        self.specialValue = specialValue # 特殊值 如'missing'或'-999999'
        self.ylabel = ylabel             # y标签的特征名称

    
    def split_assign(self,charValue,cutOffPoints):
        '''
        功能描述：运用分箱切点，将数据映射为对应的分箱点

        输入：
        charValue: 数值 （某个特征下的具体数值）
        cutOffPoints: 分割点
        specialValue：特殊值，如缺失值

        输出：
        左开右闭形式的分箱点

        管理记录：
        1. edited by 王文丹 2021/07/08
        '''

        try:
            # 变量去重
            cutOffPoints = sorted(list(set((cutOffPoints))))
            if charValue == self.specialValue:
                return 'missing'
            elif charValue <= cutOffPoints[0]:
                return '(-inf,{}]'.format(cutOffPoints[0])
            elif charValue > cutOffPoints[-1]:
                return '({},+inf]'.format(cutOffPoints[-1])
            else:
                for i in range(len(cutOffPoints)-1):
                    if cutOffPoints[i] < charValue <=  cutOffPoints[i+1]:
                        return '({},{}]'.format(cutOffPoints[i],cutOffPoints[i+1])

        except Exception as e:

            print('E_BinResult,SplitPointApplyModule,SplitPointApplyHandle,SplitPointApplyClass,split_assign',':',e)


    def split_Bin(self, dataForSplit, colForSplit, cutOffPoints):

        """
        功能描述：将分箱的切分点运用至数据集上，形成splitCharname_Bin分箱特征

        输入：
        dataForSplit: 经过数据清洗后的数据集 DataFrame
        colForSplit: 用于分箱的特征 char
        cutOffPoints: 分箱切点 list

        输出：
        dataForSplit: 附带splitCharname_Bin分箱特征的数据集

        管理记录：
        1. edited by 王文丹 2021/07/08

        """

        try:

            dataForSplit.loc[:,colForSplit+'_Bin'] = dataForSplit.loc[:,colForSplit].apply(lambda x: self.split_assign(x,cutOffPoints))
            return dataForSplit[colForSplit+'_Bin']

        except Exception as e:

            print('E_BinResult,SplitPointApplyModule,SplitPointApplyHandle,SplitPointApplyClass,split_Bin',':',e)
        
      
    def split_index(self,cutOffPoints):
        '''
        功能描述：根据数据大小，形成分箱结果groupby的索引,使得分箱结果展示按照正确的大小排序

        输入：
        cutOffPoints: 数据切分点

        输出：
        cutIndex: 根据切点形成的分箱结果groupby的索引

        管理记录：
        1. edited by 王文丹 2021/07/08
        '''
            
        try:

            cutIndex = []
            if len(cutOffPoints)<=1:
                cutIndex.append('(-inf,{}]'.format(cutOffPoints[0]))
                cutIndex.append('({},+inf]'.format(cutOffPoints[0]))
            else:
                for i in range(len(cutOffPoints)):
                    if i == 0:
                        cutIndex.append('(-inf,{}]'.format(cutOffPoints[i]))
                        cutIndex.append('({},{}]'.format(cutOffPoints[i],cutOffPoints[i+1]))

                    else:
                        if cutOffPoints[i] == cutOffPoints[-1]:
                            cutIndex.append('({},+inf]'.format(cutOffPoints[i]))
                        else: 
                            cutIndex.append('({},{}]'.format(cutOffPoints[i],cutOffPoints[i+1]))
                            
            cutIndex.append('missing')

            return cutIndex

        except Exception as e:

            print('E_BinResult,SplitPointApplyModule,SplitPointApplyHandle,SplitPointApplyClass,split_index',':',e)
    
    def split_result_bg(self,dataForSplit, colBinForSplit, cutOffPoints):

        """
        功能描述：分箱切点对应区间的好坏计算

        输入：
        dataForSplit: 经过数据清洗后的数据集 DataFrame 
        colBinForSplit: 特征名称 char
        cutOffPoints: 切分点 list

        输出：
        regroupBadGood: 分箱结果(区间对应好坏数目) DataFrame

        管理记录：
        1. edited by 王文丹 2021/07/08 
        
        """
        try:
            regroupBadGood = dataForSplit.groupby(by=colBinForSplit).agg({self.ylabel:['count','sum']})
            regroupBadGood.columns = ['All','bad']
            regroupBadGood['good'] = regroupBadGood['All']-regroupBadGood['bad']
            regroupBadGood1 = pd.DataFrame(regroupBadGood, index = self.split_index(cutOffPoints))
            regroupBadGood1 = regroupBadGood1.reset_index(level=0)
            regroupBadGood1.columns = [colBinForSplit,'All','bad','good']
            if list(regroupBadGood1[regroupBadGood1[colBinForSplit]=='missing']['All'].isnull())[0] == True:
                regroupBadGood1 = regroupBadGood1[regroupBadGood1[colBinForSplit]!='missing']
            return regroupBadGood1
        except Exception as e:
            print('E_BinResult,SplitPointApplyModule,SplitPointApplyHandle,SplitPointApplyClass,split_result_bg',':',e)

    # 把split_result 移到了E_BinResult/SplitPointApplyModule/SplitPointApplyHandle中

    def split_result(self, dataForSplit, colBinForSplit, cutOffPoints):

        """
        功能描述：分箱结果汇总计算/IV值/WOE值的映射字典

        输入：
        dataForSplit: 经过数据清洗后的数据集 DataFrame 
        colBinForSplit: 特征名称 char
        cutOffPoints: 切分点 list

        输出：
        splitResult1: 分箱结果 DataFrame 
        IV：iv值 float
        WOEDict: WOE映射字典  dict

        管理记录：
        1. edited by 王文丹 2021/07/08
        """
        try:

            splitResult = dataForSplit.groupby(by=colBinForSplit).agg({self.ylabel:['count','sum']})
            splitResult.columns = ['All','bad']
            splitResult['good'] = splitResult['All']-splitResult['bad']
            splitResult['bad_rate'] = splitResult['bad']/splitResult['All']
            splitResult['All_pcnt'] = splitResult['All']/splitResult['All'].sum()
            splitResult['good_pcnt'] = splitResult['good']/splitResult['good'].sum()
            splitResult['bad_pcnt'] = splitResult['bad']/splitResult['bad'].sum()
            splitResult['WOE'] = splitResult.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt),axis=1)
            splitResult['IV'] = splitResult.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*(np.log(x.good_pcnt*1.0/x.bad_pcnt)),axis=1)
            # 将regroup样本的索引根据分箱结果重排序
            splitResult1 = pd.DataFrame(splitResult, index = self.split_index(cutOffPoints))
            splitResult1.columns = ['All','bad','good','bad_rate','All_pcnt','good_pcnt','bad_pcnt','WOE','IV'] 
            splitResult1 = splitResult1.reset_index(level=0)
            splitResult1.columns = [colBinForSplit,'All','bad','good','bad_rate','All_pcnt','good_pcnt','bad_pcnt','WOE','IV']  
            # 生成WOE映射字典
            if list(splitResult1[splitResult1[colBinForSplit]=='missing']['All'].isnull())[0] == True:
                splitResult1 = splitResult1[splitResult1[colBinForSplit]!='missing']
            IV = splitResult1['IV'].sum()
            WOEDict = splitResult1[[colBinForSplit,'WOE']].set_index(colBinForSplit).to_dict(orient='index')   
            return (splitResult1,IV,WOEDict)
        
        except Exception as e:

            print('E_BinResult,SplitPointApplyModule,SplitPointApplyHandle,SplitPointApplyClass,split_result',':',e)

    def combine_zero(self,splitedRegroup, cutOffPoints,colBin):

        """
        功能描述：对存在单箱的坏样本数（好样本数）的切点进行合并

        输入：
        splitedRegroup：分箱结果好坏数据统计 DataFrame
        cutOffPoints：分箱切分点 list
        colBin：分箱涉及的特征名称+'_Bin' char

        输出：
        cutOffPoints：分箱切分点 list

        管理记录：
        1. edited by 王文丹 2021/07/12
            
        """

        try:

            splitedRegroup = splitedRegroup[splitedRegroup[colBin]!='missing']
            indexGoodZeros = list(splitedRegroup[splitedRegroup['good']==0].index)
            indexBadZeros = list(splitedRegroup[splitedRegroup['bad']==0].index)
            indexAllNull = list(splitedRegroup[splitedRegroup['All'].isnull()].index)
            indexOfZeros = sorted(list(set(indexGoodZeros + indexBadZeros + indexAllNull)))

            # indexOfZeros = list((splitedRegroup[splitedRegroup['bad']==0].index) | (splitedRegroup[splitedRegroup['good']==0].index))
            
            if len(indexOfZeros) > 0 :

                if len(cutOffPoints) <= 1:
                    print("{0} {1} {2}".format(colBin,"被删除","除缺失值外只有2箱且一箱占比全为好样本或坏样本"))

                else:
                    if indexOfZeros[0] == min(splitedRegroup.index):  # 即索引为0的与下一列合并
                        cutOffPoints = cutOffPoints[1:]          
                    elif indexOfZeros[0] == max(splitedRegroup.index):  # 即索引为最大的与上一列进行合并
                        cutOffPoints = cutOffPoints[:-1]          
                    else: # 其余索引与上一列进行合并, 与上一列进行合并
                        needRemovestr = float(splitedRegroup.loc[indexOfZeros[0],colBin].split(',')[1].split(']')[0])
                        cutOffPoints.remove(needRemovestr)

            return cutOffPoints

        except Exception as e:

            print('D_SplitData,SplitPointApplyModule,SplitPointApplyHandle,SplitPointApplyClass,combine_zero',':',e)

    def monotone_check(self,splitedRegroup):

        """
        功能描述：检测分箱结果是否单调

        输入：
        splitedRegroup：分箱结果（统计性描述，如好/坏）

        输出：
        badRateNotMonotone: 是否单调结果 如[True,False,True] 存在False则不单调
        
        管理记录：
        1. edited by 王文丹 2021/07/16
        
        """

        try:
            
            badRateNotMonotone = []
            if splitedRegroup['badRate'][0]>splitedRegroup['badRate'][1]:

                for i in range(len(splitedRegroup)-1):
                    badRateNotMonotone.append(float(splitedRegroup['badRate'][i]) > float(splitedRegroup['badRate'][i+1]))

            else:

                for i in range(len(splitedRegroup)-1):
                    badRateNotMonotone.append(float(splitedRegroup['badRate'][i]) < float(splitedRegroup['badRate'][i+1]))
             
            return badRateNotMonotone

        except Exception as e:

            print('D_SplitData,SplitPointApplyModule,SplitPointApplyHandle,SplitPointApplyClass,monotone_check',':',e)









                


        

