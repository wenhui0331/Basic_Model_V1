import D_SplitData
from D_SplitData.SplitPointApplyModule.SplitPointApplyHandle import SplitPointApplyClass


class EquiDistanceSplitClass():

    """
    功能描述：输出等距分箱切点

    输入：
    dataForSplit： 经过数据清洗后的数据集 DataFrame
    colForSplit：用于分箱的特征 char
    numForSplit：分箱数目 int
    specialValue：特殊值（如因缺失造成的另分一箱） 如'missing'或'-999999'

    输出：
    函数EquiDistanceSplit：输出等距分箱的切分点 function

    管理记录：
    1. edited by 王文丹 2021/07/12
    
    """
    def __init__(self,numForSplit,specialValue,ylabel,singleBadRate01):

        self.numForSplit = numForSplit   # 分箱数目 int
        self.specialValue = specialValue # 特殊值 如'missing'或'-999999'
        self.ylabel = ylabel   # 坏样本率的特征名称 如'y'
        self.singleBadRate01 = singleBadRate01    # 单箱占比为0的合并参数 0：不需要合并 1：需要合并

    def equi_distance_split(self,dataForSplit,colForSplit):

        """
        功能描述：输出等距分箱切点

        输入：
        dataForSplit： 经过数据清洗后的数据集 DataFrame
        colForSplit：用于分箱的特征 char
        numForSplit：分箱数目 int
        specialValue：特殊值（如因缺失造成的另分一箱） 如'missing'或'-999999'

        输出：
        SplitedPoint：等距分箱的切分点 list
        splitedRegroup：等距分箱的描述性结果 DataFrame
        IV: IV值 float
        WOEDict：WOE映射字典  dict

        管理记录：
        1. edited by 王文丹 2021/07/08
        
        """

        try:

            
            dataForSplit1 = dataForSplit[dataForSplit[colForSplit] != self.specialValue]# 剔除 特殊值 比如因缺失造成的另分一箱
            lowPoint = dataForSplit1[colForSplit].min() # 设置分箱的数目
            highPoint = dataForSplit1[colForSplit].max()
            # print('等距分箱最低分数',lowPoint)
            # print('等距分箱最高分数',highPoint)
            scoreInterval = round((highPoint-lowPoint)/self.numForSplit) # 根据分箱数目 设置等距分数
            # print('等距分箱分箱个数',self.numForSplit)
            # print('等距分箱分数间隔',scoreInterval)
            SplitedPoint = []  # 根据等距分数和分箱数目，生成切分点
            for i in range(self.numForSplit+1):
                SplitedPoint.append(lowPoint + i*scoreInterval)
            
            # print('初始切割点',SplitedPoint)

            # 对切分点运用类进行重命名
            SPAC = SplitPointApplyClass(self.specialValue,self.ylabel)

            # 分配切点
            dataForSplit[colForSplit+'_Bin'] = SPAC.split_Bin(dataForSplit, colForSplit, SplitedPoint)   
            # 形成初始切割点的分箱描述性结果
            splitedBG = SPAC.split_result_bg(dataForSplit,colForSplit+'_Bin',SplitedPoint)  
            # print(splitedBG)

            # 特殊处理1：循环进行切点合并，使得每个分箱不全为好样本或者坏样本
            # 特殊处理1：循环进行切点合并，使得每个分箱不全为好样本或者坏样本
            # 特殊处理1：循环进行切点合并，使得每个分箱不全为好样本或者坏样本
            splitedBG = splitedBG[splitedBG[colForSplit+'_Bin']!='missing']
            indexGoodZeros = list(splitedBG[splitedBG['good']==0].index)
            indexBadZeros = list(splitedBG[splitedBG['bad']==0].index)
            indexOfZeros = sorted(list(set(indexGoodZeros + indexBadZeros)))
            # indexOfZeros = list((splitedBG[splitedBG['bad']==0].index) | (splitedBG[splitedBG['good']==0].index))
            while len(indexOfZeros) > 0 and self.singleBadRate01 == 1 and len(SplitedPoint) > 1 : 
                SplitedPoint = SPAC.combine_zero(splitedBG, SplitedPoint,colForSplit+'_Bin')
                # 重新分配切点
                dataForSplit[colForSplit+'_Bin'] = SPAC.split_Bin(dataForSplit, colForSplit, SplitedPoint)   
                # 重新形成描述性结果        
                splitedBG = SPAC.split_result_bg(dataForSplit, colForSplit+'_Bin', SplitedPoint)  
                splitedBG = splitedBG[splitedBG[colForSplit+'_Bin']!='missing']
                indexOfZeros = list((splitedBG[splitedBG['bad']==0].index) | (splitedBG[splitedBG['good']==0].index))
            # print('最终切割点',SplitedPoint)
            return SplitedPoint

        except Exception as e:

            print('D_SplitData,EquiDistanceSplitModule,EquiDistanceSplitHandle,EquiDistanceSplitClass,equi_distance_split',':',e)

