import numpy as np
import pandas as pd
from D_SplitData.ChiMergeSplitModule.ChiMergeSplitHandle import ChiMergeSplitClass
from D_SplitData.SplitPointApplyModule.SplitPointApplyHandle import SplitPointApplyClass
from D_SplitData.EquiFrequentSplitModule.EquiFrequentSplitHandle import EquiFrequentSplitClass


class TotalSplitApplyClass():

    """
    功能描述：整体分箱后的切点运用函数

    输入：
    oldBinSplitDict: 通过分箱计算出的原始切分点 dict 格式{'a':[4,5,6],'a1':[41,51,61]}
    needModifyDict： 需要调整的分箱切点 dict 格式{'a':[4,5,6],'a1':[41,51,61]}

    输出
    函数manual_adjustment_splitpoint：手工调整分箱切点
    函数split_all_apply：分箱运用函数，出最后的计算结果(支持手动分箱，手工调整切点)如WOE/IV/统计性描述结果等

    管理记录：
    1. edited by 王文丹 2021/07/19
    
    """
    def __init__(self,specialValue, ylabel,cutParameters,binMethod,needModifyDict):

        self.specialValue = specialValue # 特殊值 如'missing'或'-999999'
        self.ylabel = ylabel             # y标签的特征名称
        self.binMethod = binMethod       # 分箱方法 如卡方分箱'chiCutMethod'
        self.cutParameters = cutParameters  #分箱参数 字典格式 如卡方分箱 {'最小单箱占比':0.05,'卡方值置信度':0.95, '卡方值自由度':4, '是否单调':1, '初始分箱数':50}
                                    # 其中 minBinPcnt为最小单箱占比 sigLevel为卡方值的置信度 sigLevel为卡方值的自由度 badRateMonoton为是否单调 'firstCutNum'分箱的初始切分箱数
        self.needModifyDict = needModifyDict  # 需要被调整的切分点 格式为{'a':[1,2,3],'b':[3,4,5]}

    def manual_adjustment_splitpoint(self,oldBinSplitDict, needModifyDict):

        """
        功能描述：手工调整分箱切点

        输入：
        oldBinSplitDict: 通过分箱计算出的原始切分点 dict 格式{'a':[4,5,6],'a1':[41,51,61]}
        needModifyDict： 需要调整的分箱切点 dict 格式{'a':[4,5,6],'a1':[41,51,61]}

        输出：
        oldBinSplitDict：已调整部分切点的分箱切点集合 dict 格式{'a':[4,5,6],'a1':[41,51,61]}

        管理记录：
        1. edited by 王文丹 2021/07/19
        """ 
        try:

            if len(needModifyDict) > 0:

                for i in needModifyDict:
                    oldBinSplitDict[i] = needModifyDict[i]  # 字典替换
            else:
                return oldBinSplitDict

            return oldBinSplitDict
        
        except Exception as e:

            print('E_BinResult,TotalSplitApplyModule,TotalSplitApplyHandle,TotalSplitApplyClass,manual_adjustment_splitpoint',':',e)


    def split_all_apply(self,dataForSplit, allFeature):

        """
        功能描述：分箱运用函数，出最后的计算结果(支持手动分箱，手工调整切点)

        输入：
        dataForSplit: 用于分箱的数据集合 DataFrame
        allFeature: 全部需要分箱的变量集合 list
        binMethod: 离散性变量集合 list
        cutParameters: 分箱涉及的参数 

        输出：
        dataForSplit：完成分箱后的数据集合 需要分箱的变量增加data[col_Bin]列 DataFrame
        ivSumSplitedSort：记录分箱完成后的IV值(按IV值从大到小排序) DataFrame
        splitedPointDictSort： 记录分箱完成后的分位点(按IV值从大到小排序) Dict
        regroupSplitedDictSort： 记录分箱完成后的统计性描述groupby结果(按IV值从大到小排序) Dict
        WOEBinDictSort: 记录分箱完成后的WOE映射结果(按IV值从大到小排序) Dict

        管理记录：
        1. edited by 王文丹 2021/07/19
        
        """

        try:

            IVSumSplited = []   # 记录分箱完成后的IV值
            splitedPointDict = {}   # 记录分箱完成后的分位点
            regroupSplitedDict = {} # 记录分箱完成后的统计性描述groupby结果
            WOEBinDict = {} # 记录分箱完成后的WOE映射结果
            splitedPointDictSort= {} # 记录分箱完成后的分位点(按IV值排序)
            regroupSplitedDictSort = {} # 记录分箱完成后的统计性描述groupby结果(按IV值排序)
            WOEBinDictSort = {} # 记录分箱完成后的WOE映射结果(按IV值排序)
            num = 0  # 数值初始化

            # 检查参数是否正常配置，否则抛出错误
            if self.binMethod == 'chiCutMethod':
                try:
                    # 最小单箱占比
                    minBinPcnt = self.cutParameters['最小单箱占比'] 
                    # 卡方值的置信度
                    sigLevel = self.cutParameters['卡方值置信度']
                    # 卡方值的自由度
                    nFree = self.cutParameters['卡方值自由度'] # 一般自由度为分箱箱数-1
                    # 是否单调 1：单调 0：非单调
                    badRateMonotone = self.cutParameters['是否单调']
                    # 分箱的初始切分箱数
                    firstCutNum = self.cutParameters['初始分箱数']
                    # 重定义卡方分箱的类
                    ChiMerge = ChiMergeSplitClass(self.specialValue,self.ylabel,minBinPcnt,sigLevel,nFree,badRateMonotone,firstCutNum)
                    # 重定义等频分箱的类
                    EFQS = EquiFrequentSplitClass(nFree+1, self.specialValue, self.ylabel, 1)  # 此处强制单箱全为好（坏）样本，则上下自动合并
                    # 对切分点运用类进行重命名
                    SPAC = SplitPointApplyClass(self.specialValue,self.ylabel)

                except Exception as e:

                    print('E_BinResult,TotalSplitApplyModule,TotalSplitApplyHandle,TotalSplitApplyClass,split_result卡方值的参数未定义',':',e)

            for colForSplit in allFeature:

                # # 变量初始化
                # colSplitedPoint = []  # 切点的初始化
                # splitedRegroup = [] # 分箱统计性描述的初始化
                # IV = [] # IV值得初始化
                # WOEDict = [] # WOE字典得初始化

                try:   # 此处抓住分箱时，抛出问题的分箱字段

                    num = num + 1
                    print('第',num,'个特征',colForSplit,'正在分箱中.....')

                    # 确保数据都为数值型
                    dataForSplit[colForSplit] = dataForSplit[colForSplit].map(lambda x: float(x))

                    # 输出卡方分箱的最终切点，若是有预先调整切点，则以调整的切点为准
                    if colForSplit in list(self.needModifyDict):
                        colSplitedPoint = self.needModifyDict[colForSplit] 
                    else:
                        # 后续可加其他分箱的方法，此处是添加不同分箱方法的入口
                        # 后续可加其他分箱的方法，此处是添加不同分箱方法的入口
                        # 后续可加其他分箱的方法，此处是添加不同分箱方法的入口
                        if self.binMethod == 'chiCutMethod':  # 离散型变量和连续性变量一起参与卡方分箱（因为所有变量已经数值化）
                            colSplitedPoint = ChiMerge.chi_merge_split(dataForSplit,colForSplit)[0]
                        # elif self.binMethod = 'ksCutMethod':
                        #     colSplitedPoint = ksMerge.ks_merge_split(dataForSplit,colForSplit)[0]

                    # 分箱最终结果展示
                    dataForSplit[colForSplit+'_Bin'] = SPAC.split_Bin(dataForSplit, colForSplit, colSplitedPoint)
                    (splitedRegroup,IV,WOEDict) = SPAC.split_result(dataForSplit, colForSplit+'_Bin', colSplitedPoint)
                    # 存储切点
                    splitedPointDict[colForSplit] = colSplitedPoint
                    # 存储IV值
                    IVSumSplited.append(IV)
                    # 存储分箱描述性结果
                    regroupSplitedDict[colForSplit] = splitedRegroup   
                    # 存储分箱的WOE映射
                    WOEBinDict[colForSplit] = WOEDict

                    # print(colSplitedPoint)
                    # print(splitedRegroup)
                
                except Exception as e:
                    print('E_BinResult,TotalSplitApplyModule,TotalSplitApplyHandle,TotalSplitApplyClass,split_all_apply,',':',colForSplit)
                

            # 统一输出排序形式（统一先以IV值排序进行统一输出）
            IVSumSplited = pd.DataFrame([allFeature,IVSumSplited]).T
            IVSumSplited.columns = ['特征名称','IV值']
            IVSumSplitedSort = IVSumSplited.sort_values(by='IV值',ascending=False)

            for colForSplit in IVSumSplitedSort['特征名称']:
                splitedPointDictSort[colForSplit+'_Bin'] = splitedPointDict[colForSplit]
                regroupSplitedDictSort[colForSplit+'_Bin'] = regroupSplitedDict[colForSplit]
                WOEBinDictSort[colForSplit+'_Bin'] = WOEBinDict[colForSplit]

            return (dataForSplit,IVSumSplitedSort,splitedPointDictSort,regroupSplitedDictSort,WOEBinDictSort)

        except Exception as e:

            print('E_BinResult,TotalSplitApplyModule,TotalSplitApplyHandle,TotalSplitApplyClass,split_all_apply,',':',e)

        

            

            

                



















  



