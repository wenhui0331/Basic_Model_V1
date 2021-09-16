from C_PreprocessData.TransformValueModule.TransformValueHandle import TransformValueClass
import pandas as pd
import numpy as np
import C_PreprocessData
import A_GetData

class ZPreprocessCallClass():

    """
    功能描述：缺失值处理C_PreprocessData的总调用函数

    输入：
    thresholdMissingDele：缺失删除阈值 float
    styleFillingVar：缺失填充方式 str ('1':统一填充为missing,'2':中位数,'3':均值,'4':众数)
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
    函数Preprocess_all_Call：数据清洗包的总调用函数
 

    管理记录：
    1. edited by 王文丹 2021/07/02 
    
    """

    def __init__(self,thresholdMissingDele,styleFillingVar,specialValue,thresholdSingleDrop,listNeedTransform,ylabel,listDropRepeat,objectNeedRemove,NumNeedRemove,pathSaved):

        print('_______________________数据清洗总调用———————————————————————————————')

        self.thresholdMissingDele = thresholdMissingDele  #缺失删除阈值 float 默认0.9
        self.styleFillingVar = styleFillingVar #缺失填充方式 int ('1':统一填充为missing,'2':中位数,'3':均值,'4':众数)
        self.specialValue = specialValue #缺失值填充值 数值型变量float (如一般统一填充为-999999)
        self.thresholdSingleDrop = thresholdSingleDrop  #单一值处理阀值 float 默认0.9
        self.listNeedTransform = listNeedTransform  #需要数据类型转换的字符型变量 list
        self.ylabel = ylabel   #坏样本率对应的特征名称 （如：'y')
        self.listDropRepeat = listDropRepeat # 确认数据集是否重复的字段列表 list
        self.objectNeedRemove = objectNeedRemove # 需要被删除的字符型变量
        self.NumNeedRemove = NumNeedRemove # 需要被删除的数值型变量
        self.pathSaved = pathSaved # 数据清洗结果的存储路径 (如：'D:/1.csv' str)
        

    def Preprocess_all_Call(self,dataForPreprocess,dataDict,nameMappingLeft,nameMappingRight):

        """
        功能描述：数据清洗包的总调用函数

        输入：
        dataForPreprocess: 以备数据清洗的数据集合 DataFrame
        thresholdMissingDele：缺失删除阈值 float
        styleFillingVar：缺失填充方式 str ('1':统一填充为missing,'2':中位数,'3':均值,'4':众数)
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
        dataForPreprocess: 经过一系列的数据清洗，清洗好的数据 DataFrame
        dataTypesAll: 数据的统计学描述结果(涵盖数值转换前后的分位数、缺失情况等) DataFrame
        listColRemains：经过一系列的数据清洗，留下的用于建模的数据 list


        管理记录：
        1. edited by 王文丹 2021/07/02
        
        """
        colDevelopment = pd.DataFrame()  # 记录变量演变情况字典
        filledStyle = pd.DataFrame()     # 记录填充方式字典
        # 1. 读取数据（可将读取数据放进模型建立当中, 此处为纯净的数据清洗)
        # print('___________________________1.读取数据_____________________________')
        # dataForPreprocess = A_GetData.DocumentsInputClass().document_input(dataPath,dataStyle) # 读取数据

        # 2. 字典转换
        print('___________________________1. 字典转换_____________________________')
        if dataDict is not None:  # 字典不为空才需要转换
            dataForPreprocess = C_PreprocessData.DictTransformClass().dict_transform(dataForPreprocess,dataDict,nameMappingLeft,nameMappingRight) # 字典转换数据

        # 3. 查看数据情况，输出字符型变量与数值型变量
        print('___________________________2. 查看数据情况，输出字符型变量与数值型变量_____________________________')
        (dataTypesBefore,listObject,listNum) =  TransformValueClass().output_var_type(dataForPreprocess)
        print( '{0}{1}' . format ( '初始字符型变量：',len(listObject ) ) )
        print( '{0}{1}' . format ( '初始数值型变量：',len(listNum ) ) )
        print( '{0}{1}' . format ( '初始字符型变量：',listObject) )
        print( '{0}{1}' . format ( '初始数值型变量：',listNum ) )

        # 4. 进行变量移除（某些变量不需要参与建模分析）
        print('_____________________________3. 进行变量移除_____________________________')

        for col in self.objectNeedRemove:  # 字符型变量移除
            if col != 'N':
                listObject.remove(col)
                colDevelopment.loc[col,'变量演变历史'] = '1.不参与数据分析'

        for col in self.NumNeedRemove:     # 数值型变量移除
            if col != 'N':
                listNum.remove(col)
                colDevelopment.loc[col,'变量演变历史'] = '1.不参与数据分析'

        print( '{0}{1}' . format ( '保留的字符型变量个数：',len(listObject)))
        print( '{0}{1}' . format ( '保留的数值型变量个数：',len(listNum)))
        print( '{0}{1}' . format ( '保留的符型变量：',listObject))
        print( '{0}{1}' . format ( '保留的数值型变量：',listNum))

        listColRemains = listNum + listObject                   ##  留存分析字段

        # 5. 进行字符型变量转换（字符型变量转换为数值型变量）
        print('_____________________________4. 输出需要数值转换对应的映射字典（按坏样本率排序）_____________________________')
        dictObjectMapping = C_PreprocessData.TransformValueClass().transform_object_dict(dataForPreprocess,self.listNeedTransform,self.ylabel)
        

        # 6. 字符型变量转换为数值型变量
        print('______________________________________5. 字符型变量转换为数值型变量__________________________________________')
        dataForPreprocess = C_PreprocessData.TransformValueClass().transform_object_type(dataForPreprocess,self.listNeedTransform,dictObjectMapping)

        # 7. 观测数据缺失情况
        
        print('______________________________________6. 数值替换完成后的变量缺失情况__________________________________________')
        missValueClass = C_PreprocessData.MissValueClass(self.thresholdMissingDele,self.styleFillingVar,self.specialValue)
        dataTypesAfter = missValueClass.missing_cal(dataForPreprocess)

        # 8. 将缺失率太高的变量删除
        print('______________________________________7. 数值替换完成后的变量缺失情况__________________________________________')
        (colMissingOverThreshold,remainColumns) = missValueClass.missing_delete_var(dataForPreprocess)

        for col in colMissingOverThreshold:  # 缺失率太高变量移除
            if col != 'N':
                listColRemains.remove(col)
                colDevelopment.loc[col,'变量演变历史'] = '2.缺失率>='+str(self.thresholdMissingDele)+'被移除'
        
        # 9. 将缺失值进行替换
        print('_______________________________________________8. 缺失值填充___________________________________________________')
        dataForPreprocess = missValueClass.missing_fill_var(dataForPreprocess,listColRemains)


        # 10. 重复值空行处理
        print('_______________________________________________9. 重复值空行处理___________________________________________________')
        repeatValueClass = C_PreprocessData.RepeatValueClass(self.thresholdSingleDrop)
        dataForPreprocess = repeatValueClass.repeat_drop_value(dataForPreprocess,self.listDropRepeat)

        # 11. 单一值处理
        print('_______________________________________________10. 单一值处理___________________________________________________')
        (colSingleDele,dataForPreprocess) = repeatValueClass.single_drop_value(dataForPreprocess,listColRemains)

        for col in colSingleDele:  # 单一值变量移除
            if col != 'N':
                listColRemains.remove(col)   #参与数据分析的字段
                colDevelopment.loc[col,'变量演变历史'] = '3.单一值>='+str(self.thresholdSingleDrop)+'被移除'

        # 12. 数据清洗结果输出
        print('_______________________________________________11. 数据清洗结果输出___________________________________________________')

            # 12.1 输出字符串转换为数值型变量的文本
        filename = open(self.pathSaved+'/5.字符型字典转换记录.txt','w')#dict转txt
        for k,v in dictObjectMapping.items():
            filename.write(k+':'+str(v))
            filename.write('\n')
        filename.close()

            # 12.2 输出处理好的数据集合
        dataForPreprocess.to_csv(self.pathSaved+'/处理后数据集.csv')

            # 12.3 数据处理过程输出
        dataTypesBefore.columns = ['变量名称','数值替换前_数据类型','数值替换前_count','数值替换前_mean','数值替换前_std','数值替换前_min','数值替换前_25%','数值替换前_50%','数值替换前_75%','数值替换前_max']

        dataTypesAfter.columns = ['变量名称','数值替换后_数据类型','数值替换后_count','数值替换后_mean','数值替换后_std','数值替换后_min','数值替换后_25%','数值替换后_50%','数值替换后_75%','数值替换后_max','数值替换后_缺失值个数','数值替换后_缺失率']

        
        dataTypesAll = pd.merge(dataTypesBefore, dataTypesAfter, on='变量名称',how='left')
        colDevelopment = colDevelopment.reset_index(level=0)
        colDevelopment.columns = ['变量名称','变量演变历史']
        dataTypesAll = pd.merge(dataTypesAll, colDevelopment,on='变量名称', how='left')
        for col in dictObjectMapping:
            filledStyle.loc[col,'转换字典'] = str(dictObjectMapping[col])
        
        if filledStyle.shape[0] > 0:    # 当无字符型变量转换为数值型变量的特殊处理
            filledStyle = filledStyle.reset_index(level=0)
            filledStyle.columns = ['变量名称','转换字典']
            dataTypesAll = pd.merge(dataTypesAll, filledStyle,on='变量名称', how='left')
            
        # dataTypesAll.to_excel(self.pathSaved+'/2.数据处理过程.xlsx')   # 可以放着建模的时候一起打印

        return (dataForPreprocess,listColRemains,dataTypesAll)

