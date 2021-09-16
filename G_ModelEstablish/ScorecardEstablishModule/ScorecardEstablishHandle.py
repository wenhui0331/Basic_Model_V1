import pandas as pd
import numpy as np
import xlwt
import xlrd
import openpyxl
from A_GetData.DocumentsInputModule.DocumentsInputHandle import DocumentsInputClass
from C_PreprocessData.ZPreprocessCallModule.ZPreprocessCallHandle import ZPreprocessCallClass
from D_SplitData.SplitPointApplyModule.SplitPointApplyHandle import SplitPointApplyClass
from E_BinResult.TotalSplitApplyModule.TotalSplitApplyHandle import TotalSplitApplyClass
from E_BinResult.TotalSplitApplyModule.WoeTransformHandle import WoeTransformClass
from F_SelectChar.FeaSelectSummaryModule.ZFeaSelectCallHandle import FeaSelectProcessClass
from G_ModelEstablish.ScorecardEstablishModule.LinearFitWithScoreScaleHandle import LinearFitWithScoreScaleClass
from I_ModelDisplay.DisplayFormModule.DisplayFormHandle import DisplayFormClass
from I_ModelDisplay.ScorecardDisplayModule.ScorecardDisplayHandle import ModelDisplayClass


class ScorecardEstablishClass():

    """
    功能描述：整体建模流程函数-建模流程主函数

    输入：
    dataPath: 数据所在的路径，如'D:/1.csv' str
    dataStyle: 'csv'/'txt'/'excel'  str

    数据清洗时用到的变量
    thresholdMissingDele：缺失删除阈值 float
    styleFillingVar：缺失填充方式 int (1:中位数,2:众数,3:均值,4:统一填充missing)
    thresholdSingleDrop：单一值处理阀值 float
    listNeedTransform：需要数据类型转换的字符型变量 list
    ylabe：坏样本率对应的特征名称 （如：'y')
    listDropRepea：确认数据集是否重复的字段列表 list
    objectNeedRemove：需要被删除的字符型变量 list
    NumNeedRemove：需要被删除的数值型变量 list
    dataDict: 字典数据 DataFrame
    nameMappingLeft: 字典中的变量名称（一般为英文,'变量名'）
    nameMappingRight: 字典中的变量名称解释（一般为中文释义,'解释'）
    listNeedTransform: 需要进行数值转换的变量

    输出：

    管理记录：
    1. edited by 王文丹 2021/07/19
    2. altered by 马文会 2021/09/15

    """
    def __init__(self,thresholdMissingDele,styleFillingVar,specialValue,thresholdSingleDrop,listNeedTransform,ylabel,listDropRepeat,objectNeedRemove,NumNeedRemove,pathSaved,maxNumBox,binMethod,cutParameters,needModifyDict,ivThreshold,xgbSelectNum,psiThreshold,corrThreshold,vifThreshold,pThreshold,selection,sle,sls,includes,baseScore,pdo,numForSplit,preThreshold):

        self.thresholdMissingDele = thresholdMissingDele  #缺失删除阈值 
        self.styleFillingVar = styleFillingVar #缺失填充方式 int ('1':统一填充为missing,'2':中位数,'3':均值,'4':众数)
        self.specialValue = specialValue #特殊值填充 数值型变量 (如-999999)
        self.thresholdSingleDrop = thresholdSingleDrop  #单一值处理阀值 float
        self.listNeedTransform = listNeedTransform  #需要数据类型转换的字符型变量 list
        self.ylabel = ylabel   #坏样本率对应的特征名称 （如：'y')
        self.listDropRepeat = listDropRepeat # 确认数据集是否重复的字段列表 list
        self.objectNeedRemove = objectNeedRemove # 需要被删除的字符型变量
        self.NumNeedRemove = NumNeedRemove # 需要被删除的数值型变量
        self.pathSaved = pathSaved # 数据清洗结果的存储路径 (如：'D:/1.csv' str)
        self.maxNumBox = maxNumBox # 最大分箱数目
        self.binMethod = binMethod # 分箱方法 如卡方分箱'chiCutMethod'
        self.cutParameters = cutParameters # 分箱的参数设置 分箱参数 字典格式 如卡方分箱 {'最小单箱占比':0.05,'卡方值置信度':0.95, '卡方值自由度':4, '是否单调':1, '初始分箱数':50}
        self.needModifyDict = needModifyDict # 需要被调整的切分点 格式为{'a':[1,2,3],'b':[3,4,5]}
        # 以下是数据筛选的参数
        self.ivThreshold = ivThreshold #IV值筛选阈值 默认参数为：0.02
        self.xgbSelectNum = xgbSelectNum #xgboost筛选阈值 默认参数为：None
        self.psiThreshold=psiThreshold #psi稳定系数筛选阈值 默认参数为：0.1
        self.corrThreshold=corrThreshold  #相关系数筛选阈值 默认参数为0.6
        self.vifThreshold=vifThreshold #vif膨胀系数筛选阈值 默认为10
        self.pThreshold=pThreshold #置信度筛选阈值 默认0.05
        self.selection=selection #逐步回归的方式，默认为逐步回归
        self.sle=sle #特征进入模型的标准即统计意义水平值，默认0.05
        self.sls=sls #特征在模型中保留的标准即统计意义水平值，默认0.05
        self.includes=includes #逐步回归时强制进入模型的特征 
        # 以下是评分卡分数的参数
        self.baseScore = baseScore #基础分数
        self.pdo = pdo #分数刻度

        # 增加 - 以下是model效果展示需要的参数
        self.numForSplit = numForSplit # 用于最后
        self.preThreshold = preThreshold #经验定义的坏样本的阈值 
    
        

    def scorecard_establishment(self,dataPath,dataStyle,dataDictPath,dataPathStyle,nameMappingLeft,nameMappingRight,dataSeparation):

        """
        功能描述：整体建模流程函数

        输入：
        dataPath: 数据所在的路径，如'D:/1.csv' str
        dataStyle: 'csv'/'txt'/'excel'  str
        dataDictPath: 数据字典所在的路径，如'D:/1.csv' str
        dataPathStyle：'csv'/'txt'/'excel'  str
        dataDict: 字典数据 DataFrame
        nameMappingLeft: 字典中的变量名称（一般为英文,'变量名'）
        nameMappingRight: 字典中的变量名称解释（一般为中文释义,'解释'）
        dataSeparation: 区分训练集合/测试集合/样本外集合的字典 
                        格式为{'区分类型字段':'same_type','训练集合':'Train','测试集合':'Test','样本外集合':'Oot'}


        输出：

        管理记录：
        1. edited by 王文丹 2021/07/25
        
        """

        print('模型在建立当中waitting for a moment........')

        print('步骤一：读取数据(数据集合和字典数据)')
        dataForPreprocess = DocumentsInputClass().document_input(dataPath,dataStyle) # 读取数据
        # 字典初始化 
        if dataDictPath != None:
            dataDict = DocumentsInputClass().document_input(dataDictPath,dataPathStyle)    
        else:
            dataDict = None
            

        print('步骤二：清洗数据')
        # 重定义数据清洗的类
        clearcallclass = ZPreprocessCallClass(self.thresholdMissingDele,self.styleFillingVar,self.specialValue,self.thresholdSingleDrop,self.listNeedTransform,self.ylabel,self.listDropRepeat,self.objectNeedRemove,self.NumNeedRemove,self.pathSaved)
        # 在此步骤的时候，已经转存：处理后数据/数据处理过程
        (dataForSplit,listForSplit,dataTypesAll) = clearcallclass.Preprocess_all_Call(dataForPreprocess,dataDict,nameMappingLeft,nameMappingRight)

        print('步骤三：定义出训练集合、测试集合、样本外等集合')
        sampleType = dataSeparation['区分类型字段']
        Train = dataSeparation['训练集合']
        Test = dataSeparation['测试集合']
        Oot = dataSeparation['样本外集合']
        dataForSplit = dataForSplit[dataForSplit[self.ylabel].isin([0,1])]  #框定有明确好坏标签的样本进入建模
        dataForSplitTrain = dataForSplit[dataForSplit[sampleType]==Train]
        dataForSplitTest = dataForSplit[dataForSplit[sampleType]==Test]
        dataForSplitOot = dataForSplit[dataForSplit[sampleType]==Oot]

        print('步骤四：根据最大分箱数目，定出类别型变量和数值型变量')
        #类别型变量
        categoricalFeatures = []
        #连续型变量
        numericalFeatures = []
        for var in listForSplit:
            if len(set(dataForSplit[var])) > self.maxNumBox:
                numericalFeatures.append(var)
            else:
                categoricalFeatures.append(var)
        print("连续型变量有{}个".format(len(numericalFeatures)))      
        print(numericalFeatures)
        print('离散型变量有{}个'.format(len(categoricalFeatures)))
        print(categoricalFeatures)
        
        print('步骤五：根据训练集合进行分箱，计算IV、WOE、dict等')
        # 重定义分箱主调用所在的类
        # listForSplit = ['最近一年内的查询次数','最近6个月贷款审核查询记录','与银行建立信贷关系月数', '授信总额（信用总额）', '授信余额（信用总余额）', '有信用业务的银行数', '现有贷款授信银行数']
        # dataForSelectTrain：完成分箱后的数据集合 需要分箱的变量增加data[col_Bin]列 DataFrame
        # IVSumSplitedSort：记录分箱完成后的IV值(按IV值从大到小排序) DataFrame
        # splitedPointDictSort：记录分箱完成后的分位点(按IV值从大到小排序) Dict 
        # regroupSplitedDictSort：记录分箱完成后的统计性描述groupby结果(按IV值从大到小排序) Dict
        # WOEBinDictSort：记录分箱完成后的WOE映射结果(按IV值从大到小排序) Dict
        TSAC = TotalSplitApplyClass(self.specialValue, self.ylabel,self.cutParameters,self.binMethod,self.needModifyDict)
        (dataForSelectTrain,IVSumSplitedSort,splitedPointDictSort,regroupSplitedDictSort,WOEBinDictSort) = TSAC.split_all_apply(dataForSplitTrain, listForSplit)

        # 将训练集合切点运用至测试集合结果
        dataForSplitTest = WoeTransformClass(self.specialValue, self.ylabel).apply_splitPoint_to_test(dataForSplitTest,list(splitedPointDictSort),splitedPointDictSort,WOEBinDictSort)

        IVSumSplitedTest = []
        regroupSplitedDictTest = {}
        for col in regroupSplitedDictSort:
            (splitedRegroupTest,IVTest,WOEDictTest) = SplitPointApplyClass(self.specialValue,self.ylabel).split_result(dataForSplitTest, col, splitedPointDictSort[col])
            IVSumSplitedTest.append(IVTest)
            regroupSplitedDictTest[col] = splitedRegroupTest

        # 形成测试集合对应切点的IV值
        colName = []
        for s in list(regroupSplitedDictSort):
            colName.append(s.replace('_Bin',''))
        IVSumSplitedTest = pd.DataFrame([colName,IVSumSplitedTest]).T
        IVSumSplitedTest.columns = ['特征名称','测试IV值']
        IVAll = pd.merge(IVSumSplitedSort,IVSumSplitedTest,on='特征名称',how='left')

        # 分箱结果保存
        workbook1 = openpyxl.Workbook()
        # 1. 训练集合分箱结果保存
        DisplayFormClass.write_excel_xlsx(self.pathSaved+'/1.数据分箱结果概览.xlsx','训练集合分箱结果',regroupSplitedDictSort,workbook1, style = 1) 
        # 2. 测试集合分箱结果保存
        DisplayFormClass.write_excel_xlsx(self.pathSaved+'/1.数据分箱结果概览.xlsx','测试集合分箱结果',regroupSplitedDictTest,workbook1, style = 1)
        # 3. 全部IV值保存
        DisplayFormClass.write_excel_xlsx(self.pathSaved+'/1.数据分箱结果概览.xlsx','IV值排序',{'IV':IVAll},workbook1,style = 1) 
        # 3. 数据切点存储
        filename = open(self.pathSaved+'/3.数据分箱切点.txt','w')#dict转txt
        for k,v in splitedPointDictSort.items():
            filename.write(k+':'+str(v))
            filename.write('\n')
        filename.close()
        # 4. WOE映射存储
        filename = open(self.pathSaved+'/4.WOE映射结果.txt','w')#dict转txt
        for k,v in WOEBinDictSort.items():
            filename.write(k+':'+str(v))
            filename.write('\n')
        filename.close()

        print('步骤六：去除iV为inf的变量')
        IVInfDropVar = list(IVSumSplitedSort[IVSumSplitedSort['IV值']==np.inf]['特征名称']) # 取IV值为inf的所有特征名称列表
        [regroupSplitedDictSort.pop(colIVinf+'_Bin') for colIVinf in IVInfDropVar]   # 将取值为inf的特征所在的regroup描述从字典中删除
        

        print('步骤七：对挑选IV值高的变量并对其进行WOE变换: 尽量不要让所有变量都参与WOE变换，否则会使得数据存储量过大')
        # dataForSelectTrain: 对IV值高的变量，已增加WOE转换的数据列（IV值低的变量未进行转换） DataFrame
        # ivHighVarBinName: 高IV值变量集合, 格式为['现有贷款授信银行数_Bin', '有信用业务的银行数_Bin'] list
        # ivLowVarBinName: 低IV值变量集合, 格式为['现有贷款授信银行数_Bin', '有信用业务的银行数_Bin'] list
        # ivHighVarWOEName: 高IV值变量集合, 格式为['现有贷款授信银行数_Bin_WOE', '有信用业务的银行数_Bin_WOE'] list
        # ivHighDict: 高IV值 IV映射字典 格式为{'现有贷款授信银行数_Bin': 0.1896299049533085, '有信用业务的银行数_Bin': 0.17689862310896204}
        (dataForSelectTrain,ivHighVarBinName,ivLowVarBinName,ivHighVarWOEName,ivHighDict) = WoeTransformClass(self.specialValue,self.ylabel).woe_transform(dataForSelectTrain,regroupSplitedDictSort,WOEBinDictSort,self.ivThreshold)
        
        
             
        print('步骤八：对测试集合进行变量的Bin和WOE映射')
        # dataForSelectTest: 已进行Bin和WOE映射的测试集合 #WOEBinDictSort
        dataForSelectTest =  WoeTransformClass(self.specialValue,self.ylabel).apply_splitPoint_to_test(dataForSplitTest,ivHighVarBinName,splitedPointDictSort,WOEBinDictSort)
        


        print('步骤九：变量筛选')
        # 重定义特征筛选主调用所在的类
        # 此处将self.ivThreshold 和 self.xgbSelectNum=None 都定义为None 原因在于woe变化时候，已经将IV值高的变量挑选出来
        FSPC = FeaSelectProcessClass(None,None,self.psiThreshold,self.corrThreshold,self.vifThreshold,self.pThreshold,self.selection,self.sle,self.sls,self.includes)

        (dataForLrTrain,featNocorr,psiDropVar,corrDropVar,vifDropVar,pDropVar,stepwiseDropVar,coefDropVar,resDataAll) = FSPC.feature_select(dataForSelectTrain[ivHighVarWOEName+[self.ylabel]],dataForSelectTest[ivHighVarWOEName+[self.ylabel]],self.ylabel,True,True)
       
        # 变量演变过程打印
        for col in IVInfDropVar:
            tempcol = col.replace('_Bin','').replace('_WOE','')
            dataTypesAll.loc[dataTypesAll['变量名称']==tempcol,'变量演变历史']='4.iv值为inf被删除'
        
        for col in ivLowVarBinName:
            tempcol = col.replace('_Bin','').replace('_WOE','')
            dataTypesAll.loc[dataTypesAll['变量名称']==tempcol,'变量演变历史']='5.iv值<'+str(self.ivThreshold)+'变量删除'

        for col in psiDropVar:
            tempcol = col.replace('_Bin','').replace('_WOE','') 
            dataTypesAll.loc[dataTypesAll['变量名称']==tempcol,'变量演变历史']='6.psi值>'+str(self.psiThreshold)+'变量删除'

        for col in corrDropVar:
            tempcol = col.replace('_Bin','').replace('_WOE','')
            dataTypesAll.loc[dataTypesAll['变量名称']==tempcol,'变量演变历史']='7.corr值>'+str(self.corrThreshold)+'变量删除'
        
        for col in vifDropVar:
            tempcol = col.replace('_Bin','').replace('_WOE','')
            dataTypesAll.loc[dataTypesAll['变量名称']==tempcol,'变量演变历史']='8.vif值>'+str(self.vifThreshold)+'变量删除'

        for col in pDropVar:
            tempcol = col.replace('_Bin','').replace('_WOE','')
            dataTypesAll.loc[dataTypesAll['变量名称']==tempcol,'变量演变历史']='9.置信度>'+str(self.pThreshold)+'变量删除'
        
        for col in stepwiseDropVar:
            tempcol = col.replace('_Bin','').replace('_WOE','')
            dataTypesAll.loc[dataTypesAll['变量名称']==tempcol,'变量演变历史']='10.逐步回归变量删除'
        
        for col in coefDropVar:
            tempcol = col.replace('_Bin','').replace('_WOE','')
            dataTypesAll.loc[dataTypesAll['变量名称']==tempcol,'变量演变历史']='11.正负符合不一致变量删除'
        
        for col in featNocorr:
            tempcol = col.replace('_Bin','').replace('_WOE','')
            dataTypesAll.loc[dataTypesAll['变量名称']==tempcol,'变量演变历史']='12.入模变量'

    
        # 变量删除 
        [ivHighVarWOEName.remove(col) for col in psiDropVar]   # 删除被psi筛选的变量
        [ivHighVarWOEName.remove(col) for col in corrDropVar]   # 删除被相关系数筛选的变量
        [ivHighVarWOEName.remove(col) for col in vifDropVar]   # 删除被膨胀系数筛选的变量
        [ivHighVarWOEName.remove(col) for col in pDropVar]   # 删除被置信度筛选的变量
        [ivHighVarWOEName.remove(col) for col in stepwiseDropVar]   # 删除被逐步回归筛选的变量
        [ivHighVarWOEName.remove(col) for col in coefDropVar]   # 删除正负符合不一致变量

        dataTypesAll.to_excel(self.pathSaved+'/2.数据处理过程.xlsx')

        print('步骤十：将筛选好的变量进行线性拟合,并输出评分卡分数')
        # # 重定义分数的线性拟合和分数刻度设置所在的类
        LFWSC = LinearFitWithScoreScaleClass(self.ylabel,self.baseScore,self.pdo)
        # 训练集合分数
        print('训练集合分数')

        (trainForLr,trainlrSmallPSummary) = LFWSC.linear_fit(dataForLrTrain[ivHighVarWOEName+[self.ylabel]],ivHighVarWOEName)
        (testForLr, testlrSmallPSummary) = LFWSC.linear_fit(dataForSelectTest[ivHighVarWOEName+[self.ylabel]+self.listDropRepeat],ivHighVarWOEName)
        print(trainForLr['score'].head())
        print(trainlrSmallPSummary)


        print ("步骤十一：展示模型效果")
        print ("preThreshold:",self.preThreshold)
        #编写主函数，存储对应的数据信息
        modelDisplayClass = ModelDisplayClass(self.numForSplit,self.specialValue,self.ylabel,1)
        modelDisplayClass.modelDisplayDF(trainForLr,testForLr,self.pathSaved,pre='score',preThreshold=self.preThreshold,probability='yPred')
        return (trainForLr,testForLr)



