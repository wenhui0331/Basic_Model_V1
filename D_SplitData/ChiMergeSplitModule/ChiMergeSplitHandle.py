import pandas as pd
import numpy as np
from pandas.io.formats.style import jinja2
from sklearn import metrics
from numpy import *
from D_SplitData.SplitPointApplyModule.SplitPointApplyHandle import SplitPointApplyClass
from D_SplitData.EquiFrequentSplitModule.EquiFrequentSplitHandle import EquiFrequentSplitClass
import scipy.stats as st
from time import *
 

class ChiMergeSplitClass():

    """
    功能描述：卡方分箱系列函数

    输入：
    specialValue：特殊值 如'missing'或'-999999' char
    ylabel：坏样本率的特征名称 如'y' char
    minBinPcnt：最小单箱占比 float 如0.05
    sigLevel：置信度 一般为0.95
    nFree：自由度 一般为最大分箱数-1  如5-1=4
    badRateMonotone：是否需要单调 1：需要控制单调 0：不需要控制单调
    firstCutNum：初始等频分箱箱数 如初始分箱一般设置为50
    dataForSplit：用于分箱的数据集合 DataFrame
    colForSplit: 用于分箱的特征名称 char

    输出：
    函数chi_merge_split：卡方分箱总调用函数，该函数具备功能如下：
        1. 初始分箱：等频分箱
        2. 单箱全部为好/坏样本处理（上下合并）
        3. 核心卡方合并分箱（寻找最小卡方值进行循环合并，直至最小卡方值<卡方阀值
        4. 检查单箱占比（对单箱占比过少的箱体，根据最小卡方值进行合并，直至每箱样本量超过最小阀值）
        5. 检查分箱结果是否单调，是否单调可控制（对分箱结果不单调的箱体，寻找相邻的最小卡方值进行合并，直至分箱结果单调）
        
    函数chi: 计算相邻两组的卡方值
    函数core_chi: 卡方合并分箱的核心函数（循环挑选最小卡方值进行切分点的合并）
    函数single_bin_pcnt: 检查单箱占比，当遇到单箱占比<初设最小占比, 该箱则与卡方值较小的箱子进行合并
    函数one_step_chimerge：卡方分箱函数（单步挑选最小卡方值进行切分点的合并，此处和循环判断单调性结合使用
                           即若该卡方分箱要求单调，会在核心分箱-检查单箱占比后进行单步合并，直至单调）

    

    管理记录：
    1. edited by 王文丹 2021/07/16
    
    """


    def __init__(self,specialValue,ylabel,minBinPcnt,sigLevel,nFree,badRateMonotone,firstCutNum):

        self.specialValue = specialValue # 特殊值 如'missing'或'-999999'
        self.ylabel = ylabel  # 坏样本率的特征名称 如'y'
        self.minBinPcnt = minBinPcnt # 最小单箱占比
        self.sigLevel = sigLevel # 置信度 一般为0.95
        self.nFree = nFree # 自由度 一般为最大分箱数-1
        self.badRateMonotone = badRateMonotone  # 是否需要单调 1：需要控制单调 0：不需要控制单调
        self.firstCutNum = firstCutNum # 初始等频分箱箱数 如初始分箱一般设置为50

    def chi_merge_split(self,dataForSplit,colForSplit):

        """
        功能描述：卡方分箱总调用函数，该函数具备功能如下：
                       1. 初始分箱：等频分箱
                       2. 单箱全部为好/坏样本处理（上下合并）
                       3. 核心卡方合并分箱（寻找最小卡方值进行循环合并，直至最小卡方值<卡方阀值
                       4. 检查单箱占比（对单箱占比过少的箱体，根据最小卡方值进行合并，直至每箱样本量超过最小阀值）
                       5. 检查分箱结果是否单调，是否单调可控制（对分箱结果不单调的箱体，寻找相邻的最小卡方值进行合并，直至分箱结果单调）
        输入：
        dataForSplit: 用于分箱的数据集合 DataFrame
        colForSplit: 用于分箱的特征名称 str
        specialValue：特殊值 与初始的特殊值设定保持一致，方便特殊值单分一箱  如'missing'或'-999999' 
        ylabel：坏样本率的特征名称 如'y' str
        minBinPcnt：最小单箱占比 如0.05 
        sigLevel：置信度 一般为0.95
        nFree：自由度 一般为最大分箱数-1 如5-1=4
        badRateMonotone：是否需要单调 (1：需要控制单调 0：不需要控制单调)
        firstCutNum: 初始等频分箱箱数 如初始分箱一般设置为50


        输出：
        SplitedPoint：卡方分箱输出的切点, 格式如[1,2,3]
        splitedBG：卡方分箱的分组描述性结果，如好/坏样本数，好样本率、坏样本率，WOE值、IV值等

        管理记录：
        1. edited by 王文丹 2021/07/16     
        """

        try:

            # 重命名类
            # 重命名类：切分点运用类进行重命名
            SPAC = SplitPointApplyClass(self.specialValue,self.ylabel)
            # 重命名类：等频分箱类进行重命名
            EFSC = EquiFrequentSplitClass(self.firstCutNum,self.specialValue,self.ylabel,1) # 此处的1表示需要合并单箱占比全为好样本或坏样本的箱子

            # 1. 去掉缺失值
            dataForSplitNoSpecial = dataForSplit[dataForSplit[colForSplit] != self.specialValue]

            # 2. 初始卡方值
            thresholdForChi = st.chi2.isf(1-self.sigLevel,df=self.nFree)

            # 3. 输出等频分箱的切点（中间已经把单箱占比全为好样本或坏样本的箱数进行了合并)
            SplitedPoint = EFSC.equi_freuent_split(dataForSplitNoSpecial,colForSplit)
            
            if len(SplitedPoint) == 0:
                print(colForSplit,'该特征全部为缺失值')
                return ([self.specialValue],[])
            else:
                # 4. 进入核心最小卡方分箱
                (SplitedPoint, splitedBG) = self.core_chi(dataForSplitNoSpecial,colForSplit,SplitedPoint,thresholdForChi)

                # 5. 对分箱占比不满足最小箱的 内部进行最小卡方合并
                (SplitedPoint,splitedBG) = self.single_bin_pcnt(dataForSplitNoSpecial,colForSplit,SplitedPoint)

                # 6. 检查是否满足单调性，挑选最小卡方值进行合并，直至单调
                if self.badRateMonotone > 0:  # 需要控制单调

                    splitedBG['badRate'] = splitedBG['bad']/splitedBG['All']
                    badRateIsMonotone = SPAC.monotone_check(splitedBG)  # 判断分箱的坏样本是否单调

                    while False in badRateIsMonotone and len(splitedBG) > 2:   # 2箱一定是单调的
                        # 每次进行单步合并，在这里不需要判断卡方值的限额，只挑选最小卡方合并即可
                        (SplitedPoint, splitedBG) = self.one_step_chimerge(dataForSplitNoSpecial,colForSplit,SplitedPoint) 
                        splitedBG['badRate'] = splitedBG['bad']/splitedBG['All']
                        badRateIsMonotone = SPAC.monotone_check(splitedBG)

            return (SplitedPoint,splitedBG)
        
        except Exception as e:

            print('D_SplitData,ChiMergeSplitModule,ChiMergeSplitHandle,ChiMergeSplitClass,chi_merge_split',':',e)


    def chi(self,dataBin):
        """

        功能描述：：计算相邻两组的卡方值

        输入：
        dataBin: 计算chi2值的数据集

        输出：
        chi: 卡方chi2值

        管理记录：
        1. edited by 王文丹 2021/07/16

        """
        try:

            # 计算总行数 该类别的总人数
            n = dataBin.sum(axis=1)

            # 计算列总数 该分类总人数 如坏样本总人数
            m = dataBin.sum(axis=0)

            # 计算全部总人数
            N = dataBin.sum().sum()

            # 计算卡方值
            E = np.array(mat(list(n)).T*mat(list(m)))
            C = E/N
            ChiMatrix = (np.array(dataBin)-C)**2/C  # 卡方值矩阵
            chi = float(round(ChiMatrix.sum(),8))
            return chi
        
        except Exception as e:

            print('D_SplitData,ChiMergeSplitModule,ChiMergeSplitHandle,ChiMergeSplitClass,chi',':',e)

    
    def core_chi(self,dataForSplit,colForSplit,SplitedPoint,thresholdForChi):

        """
        功能描述：卡方合并分箱的核心函数（循环挑选最小卡方值进行切分点的合并）

        输入：
        dataForSplit: 用于分箱的数据集合 DataFrame
        colForSplit: 用于分箱的特征名称 str
        SplitedPoint: 分箱的切分点 如[1,2,3]
        thresholdForChi: 卡方分箱的卡方阀值 此书可由卡方置信度和卡方自由度定st.chi2.isf(1-self.sigLevel,df=self.nFree)
        specialValue：特殊值 与初始的特殊值设定保持一致，方便特殊值单分一箱  如'missing'或'-999999' 
        ylabel：坏样本率的特征名称 如'y' str

        输出：
        SplitedPoint：卡方核心输出的切点 
        splitedBG：卡方分箱的分组描述性结果，好样本数、坏样本数、总样本数


        管理记录：
        1. edited by 王文丹 2021/07/16
         
        """
        try:

            # 排除异常值的影响
            dataForSplitNoSpecial = dataForSplit[dataForSplit[colForSplit] != self.specialValue]

            # 重命名类：切分点运用类进行重命名
            SPAC = SplitPointApplyClass(self.specialValue,self.ylabel) 

            # 分配切点
            dataForSplitNoSpecial.loc[:,colForSplit+'_Bin'] = SPAC.split_Bin(dataForSplitNoSpecial, colForSplit, SplitedPoint)
                
            # 形成初始切割点的分箱描述性结果（好坏数目计算)
            splitedBG = SPAC.split_result_bg(dataForSplitNoSpecial,colForSplit+'_Bin',SplitedPoint)

            # 最小卡方分箱合并
            while True:
                
                minChi = 1000 # 数值初始化
                for i in range(len(splitedBG)-1):
                    # 计算卡方值
                    kchi = self.chi(splitedBG.loc[i:i+1,['good','bad']]) 
                    splitedBG.loc[i,'chi'] = kchi

                    if kchi < minChi:    # 循环定下当前的最小卡方值
                        minIndex = i
                        minChi = kchi

                if minChi < thresholdForChi:

                    # minChiInterval = splitedBG.loc[minIndex+1,colForSplit+'_Bin']
                    # pointNeedRemove = float(minChiInterval.split(',',1)[0].split('(',1)[1])

                    pointNeedRemove = SplitedPoint[minIndex]
                    SplitedPoint.remove(pointNeedRemove)
                    if len(SplitedPoint) == 0:
                        SplitedPoint = [pointNeedRemove]
                        break
                                        
                else:
                    break
                # 重新计算对应的数值
                splitedBG.loc[minIndex,:] = splitedBG.loc[minIndex,:] + splitedBG.loc[minIndex+1,:]
                splitedBG = splitedBG.drop(splitedBG.index[minIndex+1])
                splitedBG.index = range(len(splitedBG))  # 重新定义索引
                splitedBG[colForSplit+'_Bin'] = SPAC.split_index(SplitedPoint)[:-1] 

            return (SplitedPoint, splitedBG)

        except Exception as e:

            print('D_SplitData,ChiMergeSplitModule,ChiMergeSplitHandle,ChiMergeSplitClass,core_chi',':',e)

    def single_bin_pcnt(self,dataForSplit,colForSplit,SplitedPoint):

        """
        功能描述：最检查单箱占比，当遇到单箱占比<初设最小占比, 该箱则与卡方值较小的箱子进行合并

        输入：
        dataForSplit: 用于分箱的数据集合 DataFrame
        colForSplit: 用于分箱的特征名称 str
        SplitedPoint: 分箱的切分点 如[1,2,3]
        minBinPcnt：最小单箱占比 float 如0.05
        specialValue：特殊值 与初始的特殊值设定保持一致，方便特殊值单分一箱  如'missing'或'-999999' 
        ylabel：坏样本率的特征名称 如'y' str

        输出：
        SplitedPoint：卡方核心输出的切点 
        splitedBG：卡方分箱的分组描述性结果，好样本数、坏样本数、总样本数

        管理记录：
        1. edited by 王文丹 2021/07/16
        
        """
        try:

            # 排除异常值的影响
            dataForSplitNoSpecial = dataForSplit[dataForSplit[colForSplit] != self.specialValue]

            # 重命名类：切分点运用类进行重命名
            SPAC = SplitPointApplyClass(self.specialValue,self.ylabel)

            # 分配切点
            dataForSplitNoSpecial[colForSplit+'_Bin'] = SPAC.split_Bin(dataForSplitNoSpecial, colForSplit, SplitedPoint)

            # 形成初始切割点的分箱描述性结果（好坏数目计算)
            splitedBG = SPAC.split_result_bg(dataForSplitNoSpecial,colForSplit+'_Bin',SplitedPoint)

            # 计算总样本数目
            splitedBG['All'] = splitedBG['good'] + splitedBG['bad']
            splitedBG['AllPcnt'] = splitedBG['All']/splitedBG['All'].sum()

            # print(splitedBG)

            if len(splitedBG[splitedBG['AllPcnt']<self.minBinPcnt]) > 0:

                while len(splitedBG[splitedBG['AllPcnt']<self.minBinPcnt]) > 0 and splitedBG.shape[0] > 2:

                    minPcntIndex = splitedBG[splitedBG['AllPcnt']<=self.minBinPcnt].index[0]

                    if minPcntIndex == 0:  # 遇到首箱的单箱占比<初设占比 则删除首个切点
                        SplitedPoint = SplitedPoint[1:]

                    elif minPcntIndex == splitedBG.shape[0]-1:  # 遇到末箱的单箱占比<初设占比 则删除最后切点
                        SplitedPoint = SplitedPoint[:-1]
                    
                    else:    # 遇到存在最小箱占比<小于初设 那么则上下找卡方值较小的值进行合并
                        chi1 = self.chi(splitedBG.loc[minPcntIndex-1:minPcntIndex,['good','bad']])
                        chi2 = self.chi(splitedBG.loc[minPcntIndex:minPcntIndex+1,['good','bad']])

                        binRemoveName = splitedBG.loc[minPcntIndex,colForSplit+'_Bin']

                        if  chi1 < chi2:
                            leftStrRemove = float(binRemoveName.split(',',1)[0].split('(',1)[1])
                            SplitedPoint.remove(leftStrRemove)

                        else:
                            rightStrRemove = float(binRemoveName.split(',',1)[1].split(']',1)[0])
                            SplitedPoint.remove(rightStrRemove)


                    # 重新计算对应的数值
                    # splitedBG.loc[minPcntIndex,:] = splitedBG.loc[minPcntIndex,:] + splitedBG.loc[minPcntIndex+1,:]
                    # splitedBG = splitedBG.drop(splitedBG.index[minPcntIndex+1])
                    # splitedBG.index = range(len(splitedBG))  # 重新定义索引
                    # splitedBG[colForSplit+'_Bin'] = SPAC.split_index(SplitedPoint)[:-1]  # 重新分配切点

                
                    # 分配切点
                    dataForSplitNoSpecial[colForSplit+'_Bin'] = SPAC.split_Bin(dataForSplitNoSpecial, colForSplit, SplitedPoint)  
                    # 形成初始切割点的分箱描述性结果（好坏数目计算)
                    splitedBG = SPAC.split_result_bg(dataForSplitNoSpecial,colForSplit+'_Bin',SplitedPoint)

                    # 计算总样本数目
                    splitedBG['All'] = splitedBG['good'] + splitedBG['bad']
                    splitedBG['AllPcnt'] = splitedBG['All']/splitedBG['All'].sum()

            return (SplitedPoint,splitedBG)

        except Exception as e:

            print('D_SplitData,ChiMergeSplitModule,ChiMergeSplitHandle,ChiMergeSplitClass,single_bin_pcnt',':',e)


    def one_step_chimerge(self,dataForSplit,colForSplit,SplitedPoint):
        """
        功能描述：检查分箱结果是否单调，是否单调可控制（对分箱结果不单调的箱体，寻找相邻的最小卡方值进行合并，直至分箱结果单调）

        输入：
        dataForSplit: 用于分箱的数据集合 DataFrame
        colForSplit: 用于分箱的特征名称 str
        SplitedPoint: 分箱的切分点 如[1,2,3]
        badRateMonotone：是否需要单调 1：需要控制单调 0：不需要控制单调
        specialValue：特殊值 与初始的特殊值设定保持一致，方便特殊值单分一箱  如'missing'或'-999999' 
        ylabel：坏样本率的特征名称 如'y' str

        输出：
        SplitedPoint：卡方核心输出的切点 
        splitedBG：卡方分箱的分组描述性结果，好样本数、坏样本数、总样本数

        管理记录：
        1. edited by 王文丹 2021/07/16

        
        """
        try:
            # 排除异常值的影响
            dataForSplitNoSpecial = dataForSplit[dataForSplit[colForSplit] != self.specialValue]

            # 重命名类：切分点运用类进行重命名
            SPAC = SplitPointApplyClass(self.specialValue,self.ylabel) 

            # 单步最小卡方分箱合并 非循环
            # 单步最小卡方分箱合并 非循环
            # 单步最小卡方分箱合并 非循环

            # 分配切点
            dataForSplitNoSpecial.loc[:,colForSplit+'_Bin'] = SPAC.split_Bin(dataForSplitNoSpecial, colForSplit, SplitedPoint)
        
            # 形成初始切割点的分箱描述性结果（好坏数目计算)
            splitedBG = SPAC.split_result_bg(dataForSplitNoSpecial,colForSplit+'_Bin',SplitedPoint)
            minChi = 1000 # 数值初始化

            for i in range(len(splitedBG)-1):
                # 计算卡方值
                kchi = self.chi(splitedBG.loc[i:i+1,['good','bad']]) 
                splitedBG.loc[i,'chi'] = kchi
                
                if kchi < minChi:    # 循环定下当前的最小卡方值
                    minIndex = i
                    minChi = kchi

            # minChiInterval = splitedBG.loc[minIndex+1,colForSplit+'_Bin']
            # pointNeedRemove = float(minChiInterval.split(',',1)[0].split('(',1)[1])
            pointNeedRemove = SplitedPoint[minIndex]
            SplitedPoint.remove(pointNeedRemove)
            if len(SplitedPoint) == 0:
                SplitedPoint = [pointNeedRemove]

            # 重新计算对应的数值
            splitedBG.loc[minIndex,:] = splitedBG.loc[minIndex,:] + splitedBG.loc[minIndex+1,:]
            splitedBG = splitedBG.drop(splitedBG.index[minIndex+1])
            splitedBG.index = range(len(splitedBG))  # 重新定义索引
            splitedBG[colForSplit+'_Bin'] = SPAC.split_index(SplitedPoint)[:-1] 

            # # 最后分配切点
            # dataForSplitNoSpecial[colForSplit+'_Bin'] = SPAC.split_Bin(dataForSplitNoSpecial, colForSplit, SplitedPoint)
            # # 最后形成初始切割点的分箱描述性结果（好坏数目计算)
            # splitedBG = SPAC.split_result_bg(dataForSplitNoSpecial,colForSplit+'_Bin',SplitedPoint)
            return (SplitedPoint, splitedBG)
        
        except Exception as e:

            print('D_SplitData,ChiMergeSplitModule,ChiMergeSplitHandle,ChiMergeSplitClass,one_step_chimerge',':',e)

    









            

            






































