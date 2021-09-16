import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,roc_auc_score


# 确认路径
import os
import sys
sys.path.append(os.getcwd())

from D_SplitData import EquiDistanceSplitClass,EquiFrequentSplitClass
from H_ModelEvaluate.ModelEffectModule import ModelEffectClass


class ModelDisplayClass(object):

    """
    功能： 评分卡评价指标计算
    

    输入：
    numForSplit：分箱数目 int
    specialValue ： 特殊值 如'missing'或'-999999'
    ylabel ：坏样本率的特征名称： 如'y'
    singleBadRate01 ：单箱占比为0的合并参数 0：不需要合并 1：需要合并
    
    输出：
    函数：group_by_list 训练集、测试集的评分卡分布
    函数：confusion_matrix 混淆矩阵


    维护记录：
    1. created by mawenhui  2021/09/13


    """


    def __init__(self,numForSplit,specialValue,ylabel,singleBadRate01):

        '''
        管理记录：
        1、created by mawenhui 2021/09/13，参照文丹的等距函数
        '''

        self.numForSplit = numForSplit   # 分箱数目 int
        self.specialValue = specialValue # 特殊值 如'missing'或'-999999'
        self.ylabel = ylabel   # 坏样本率的特征名称 如'y'
        self.singleBadRate01 = singleBadRate01    # 单箱占比为0的合并参数 0：不需要合并 1：需要合并


    def group_by_list(self,dataInitial,colForSplit,method = 'EquiDistance'):
        '''
        功能：按照等频/等距对输出的score值进行分组，统计各个组下的用户数量，好坏样本数量等指标

        输入：
        dataInitial：数据集 dataframe
        colForSplit：模型预测得分列名称 string
        method: 区分是按照等频/等距分组字段，默认值-EquiDistance，或者EquiFrequent

        输出:
        resultdf:按照每一组下，总样本个数，好样本个数，ks值、lift等

        管理记录：
        1、created by mawenhui 2021/09/13
        '''

        try:
            print(self.numForSplit,self.specialValue,self.ylabel,self.singleBadRate01,dataInitial.shape,colForSplit)
            if method == 'EquiDistance':
                group_split_list = EquiDistanceSplitClass(self.numForSplit,self.specialValue,self.ylabel,self.singleBadRate01).equi_distance_split(dataInitial,colForSplit)
            elif method == 'EquiFrequent':
                group_split_list = EquiFrequentSplitClass(self.numForSplit,self.specialValue,self.ylabel,self.singleBadRate01).equi_freuent_split(dataInitial,colForSplit)
            group_split_list.insert(0,0)
            group_split_list.append(1000)
            dataInitial['score_group'] = pd.cut(dataInitial[colForSplit],group_split_list)
            resultDf = dataInitial['yPred'].groupby([dataInitial['score_group'],dataInitial[self.ylabel]]).count().unstack().sort_index().reset_index()
            resultDf['全体用户'] = dataInitial['yPred'].groupby(pd.cut(dataInitial[colForSplit],group_split_list)).count().sort_index().reset_index()['yPred']
            resultDf = resultDf.rename(columns={0:'好用户',1:'坏用户'})
            resultDf['坏样本率'] = resultDf['坏用户'] / resultDf['全体用户']
            resultDf['全体用户_占比'] = resultDf['全体用户'] / dataInitial['yPred'].count()
            resultDf['好用户_占比'] = resultDf['好用户'] / dataInitial[dataInitial[self.ylabel] == 0]['yPred'].count()
            resultDf['坏用户_占比'] = resultDf['坏用户'] / dataInitial[dataInitial[self.ylabel] == 1]['yPred'].count()
            resultDf['好用户累计数量'] = resultDf['好用户'].cumsum()
            resultDf['坏用户累计数量'] = resultDf['坏用户'].cumsum()
            resultDf['好用户累计占比'] = resultDf['好用户累计数量'] / dataInitial[dataInitial[self.ylabel] == 0]['yPred'].count()
            resultDf['坏用户累计占比'] = resultDf['坏用户累计数量'] / dataInitial[dataInitial[self.ylabel] == 1]['yPred'].count()
            resultDf['lift'] = resultDf['坏用户_占比'] / resultDf['好用户_占比']
            bincnt = resultDf['score_group'].drop_duplicates().count()
            print ("可以分组的个数为:",bincnt)
            resultDf['KS'] = resultDf['坏用户累计占比'] - (1 / bincnt)
            return resultDf[['score_group','全体用户','好用户','坏用户','坏样本率','全体用户_占比','好用户_占比','坏用户_占比','好用户累计数量','坏用户累计数量','好用户累计占比','坏用户累计占比','lift','KS']]
        
        except Exception as e:
            print('I_ModelDisplay,ModelDisplayClass,group_by_list',':',e) 

    def confusion_matrix(self,dataInitial,probability,preThreshold=0.5):
        '''
        功能：计算混淆矩阵、混淆矩阵占比及效果提升

        输入：
        dataInitial：数据集,dataframe
        probability: 预测的坏样本概率 string
        threshold:阈值，用于标记是否是坏样本 float

        输出
        confusionMatrix：混淆矩阵 dataframe
        confusionMatrixRatio:混淆矩阵占比 dataframe
        gainMatrix:模型应用后的增益比

        管理记录：
        1、created by  mawenhui 2021/09/13
        '''

        try:
            dataInitial['y_pre'] = dataInitial[probability].map(lambda x:x >= preThreshold).astype(int)
            resultDf = dataInitial[probability].groupby([dataInitial['y_pre'],dataInitial[self.ylabel]]).count().unstack()
            resultDf.loc['总体'] = resultDf.sum(axis=0)
            resultDf['总体'] = resultDf.sum(axis=1)
            resultDf = resultDf.rename(columns = {0:'好客户',1:'坏客户'},index = {0:'预测为好的客户',1:'预测为坏的客户'})
            resultRatioDf = resultDf / dataInitial[probability].count() 
            gainDf = pd.DataFrame([],index = ['未应用模型','应用模型后'],columns = ['好客户','坏客户'])
            print (dataInitial['y_pre'].groupby(dataInitial[self.ylabel]).count().to_list())
            gainDf.loc['未应用模型'] = dataInitial['y_pre'].groupby(dataInitial[self.ylabel]).count().to_list()
            gainDf.loc['应用模型后'] = dataInitial[dataInitial['y_pre'] < 1]['y_pre'].groupby(dataInitial[dataInitial['y_pre'] < 1][self.ylabel]).count().to_list()
            gainDf['坏样本占比'] = gainDf['坏客户'] / gainDf.sum(axis = 1)
            return (resultDf,resultRatioDf,gainDf)
        except Exception as e:
            print('I_ModelDisplay,ModelDisplayClass,confusion_matrix',':',e) 

    def charge_matrix(self,datatrain=None,datatest=None,probability='yPred',threshold=0.5,pre='score',label='label'):
        '''
        功能：计算评价矩阵，包括ks、auc、gini、准确率、精确度、召回率，f1值

        输入：
        datatrain：数据集 dataframe，当只评测测试集时为none
        datatest:数据集 dataframe
        probability: 预测的坏样本概率 string
        threshold:阈值，用于标记是否是坏样本 float
        pre: 排序值，一般是百位数，内部拟定的范围 string
        label:样本标记，一般为0或者1，string

        输出
        指标的评估矩阵

        管理记录：
        1、created by  mawenhui 2021/09/13
        '''
        try:
            if datatrain is not None:
                datatrain['y_pred'] = datatrain[probability].map(lambda x:x >= threshold).astype(int)
                train_ks = ModelEffectClass().get_ks(datatrain,label,pre)
                train_auc = ModelEffectClass().get_auc(datatrain,label,pre)
                train_gini = ModelEffectClass().get_gini(datatrain,label,pre)
                train_f1= f1_score(datatrain[label],datatrain['y_pred'])
                train_accuracy = accuracy_score(datatrain[label],datatrain['y_pred'])
                train_precision = precision_score(datatrain[label],datatrain['y_pred'])
                train_recall = recall_score(datatrain[label],datatrain['y_pred'])
            else:
                pass 

            datatest['y_pred'] = datatest[probability].map(lambda x:x >= threshold).astype(int)
            test_ks = ModelEffectClass().get_ks(datatest,label,pre)
            test_auc = ModelEffectClass().get_auc(datatest,label,pre)
            test_gini = ModelEffectClass().get_gini(datatest,label,pre)
            test_f1= f1_score(datatest[label],datatest['y_pred'])
            test_accuracy = accuracy_score(datatest[label],datatest['y_pred'])
            test_precision = precision_score(datatest[label],datatest['y_pred'])
            test_recall = recall_score(datatest[label],datatest['y_pred'])
        
            chargeDf = pd.DataFrame([],columns = ['ks','auc','gini','f1','accuracy','precision','recall'])
            if datatrain is not None:
                chargeDf.loc['train'] = [train_ks,train_auc,train_gini,train_f1,train_accuracy,train_precision,train_recall]
            else:
                pass 
            
            chargeDf.loc['test'] = [test_ks,test_auc,test_gini,test_f1,test_accuracy,test_precision,test_recall]
            return chargeDf
        except Exception as e:
            print('I_ModelDisplay,ModelDisplayClass,confusion_matrix',':',e) 


    def data_distribution(self,dataInitial,plotnumForSplit,pre):
        '''
        功能：绘制相关分布图，包括训练集分布图、测试集分布图，训练集好坏样本分布图、测试集好坏样本分布图

        输入：
        dataInitial：数据集 dataframe
        plotnumForSplit： 绘制分布图时score值被拆分的个数  bigint
        pre: 横坐标轴数据，用于绘图

        输出：
        用户绘制图片的信息 dataframe

        备注：
        此代码暂时未用到

        管理记录：
        1、created by  mawenhui 2021/09/14
        '''
        try:
            group_split_list = EquiDistanceSplitClass(plotnumForSplit,self.specialValue,self.ylabel,self.singleBadRate01).equi_distance_split(dataInitial,pre)
            print ("用户分Bin的list具体值v1:",group_split_list)
            group_split_list.insert(0,dataInitial[pre].min())
            group_split_list.append(dataInitial[pre].max())
            print ("用户分Bin的list具体值v2:",group_split_list)
            dataInitial['score_group'] = pd.cut(dataInitial[pre],group_split_list)
            print (dataInitial['score_group'].map(lambda x:str(x).replace('(','').replace("]",'').split(",")).head())
            print (dataInitial['score_group'].map(lambda x:0.5*float(str(x).replace('(','').replace("]",'').split(",")[0])).head())
            dataInitial['score_avg'] = dataInitial['score_group'].map(lambda x:0.5*float(str(x).replace('(','').replace("]",'').split(",")[0]) + 0.5*float(str(x).replace('(','').replace("]",'').split(",")[1]))
            dataInitial['score_avg'] = dataInitial['score_avg'].astype(float)
            print (dataInitial['score_avg'].head(),dataInitial['score_avg'].dtypes)
            return dataInitial[['score_avg',self.ylabel,pre]]
        except Exception as e:
            print('I_ModelDisplay,ModelDisplayClass,data_distribution',':',e) 
    

    def distin_ks(self,lab_true,lab_pred):
        '''
        功能：准备绘制ks图的数据集

        输入：
        lab_true:标签数据 list
        lab_pred：预测的概率值 list

        输出：
        用户绘制ks图片的数据

        管理记录：
        1、created by  mawenhui 2021/09/14
        '''
        try:
            cnt_all=len(lab_true)
            cnt_stats=lab_true.value_counts().to_dict()
            cnt_bad,cnt_good=cnt_stats[1],cnt_stats[0]
            df_ks=pd.DataFrame({'lab':lab_true,'proba':lab_pred})
            df_ks=df_ks.sort_values(by='proba',ascending=False)
            df_ks['cum_pnt_all']=[i/cnt_all for i in list(range(1,cnt_all+1))]
            df_ks['cum_pnt_bad']=df_ks.lab.cumsum()/cnt_bad
            df_ks['cum_pnt_good']=(df_ks.lab==0).cumsum()/cnt_good
            df_ks['diff']=df_ks['cum_pnt_bad']-df_ks['cum_pnt_good']
            ks=max(df_ks['diff'])
            return ks,df_ks
        except Exception as e:
            print('I_ModelDisplay,ModelDisplayClass,distin_ks',':',e) 

    def charge_plot(self,datatrain,datatest,imagePath,score='score',probability='yPred'):
        '''
        功能：绘制相关分布图，包括训练集分布图、测试集分布图，训练集好坏样本分布图、测试集好坏样本分布图

        输入：
        datatrain:训练集数据集 dataframe
        datatest:测试集数据集 dataframe
        imagePath:需要保存的图片路径
        dfName：需要保存的图片名称
        score:用户查看分布的值 string
        probability: 预测的标记 string 

        输出：
        用户绘制ks图片的数据

        管理记录：
        1、created by  mawenhui 2021/09/14
        '''
        try:
            # print ("数据测试：",datatrain.shape,numForSplit,score)
            # train_plot = self.data_distribution(datatrain,numForSplit,score)
            # test_plot = self.data_distribution(datatest,numForSplit,score)
            # print ("plot数据格式",train_plot.shape,test_plot.shape)
            # print ("plot数据情况",train_plot.head(),test_plot.head())


            plt.rcParams['font.sans-serif'] = ['SimHei']
            # 设置显示正常字符
            plt.rcParams['axes.unicode_minus'] = False
            sns.set(font='SimHei', style='white')
            fig = plt.figure(figsize=(24,14))
            ax1 = fig.add_subplot(4,2,1)
            #ax1.patch.set_facecolor("gainsboro")
            sns.kdeplot(datatrain[score],shade=True,legend=False,label='train Score Distribution')
            ax1.set_xlabel('Score')
            ax1.legend(fontsize=7)


            # # 测试集
            ax2 = fig.add_subplot(4,2,2)
            #ax1.patch.set_facecolor("gainsboro")
            sns.kdeplot(datatest[score],shade=True,legend=True,label='test Score Distribution')
            ax2.set_xlabel('Score')
            ax2.legend(fontsize=7)

            #训练集正负样本分布情况
            ax3 = fig.add_subplot(4,2,3)
            #ax1.patch.set_facecolor("gainsboro")
            sns.kdeplot(datatrain[datatrain[self.ylabel] == 0][score],color='green',shade=True,legend=True,label='Good train Score Distribution')
            sns.kdeplot(datatrain[datatrain[self.ylabel] == 1][score],color='red',shade=True,legend=True,label='Bad train Score Distribution')
            ax3.set_xlabel('Score')
            ax3.legend(fontsize=7)

            #测试集正负样本分布情况
            ax4 = fig.add_subplot(4,2,4)
            #ax1.patch.set_facecolor("gainsboro")
            sns.kdeplot(datatest[datatest[self.ylabel] == 0][score],color='green',shade=True,legend=True,label='Good test Score Distribution')
            sns.kdeplot(datatest[datatest[self.ylabel] == 1][score],color='red',shade=True,legend=True,label='Bad test Score Distribution')
            ax4.set_xlabel('Score')
            ax4.legend(fontsize=7)

            # 绘制auc图

            tpr,fpr,threshold = roc_curve(datatrain[self.ylabel],datatrain[probability]) 
            train_AUC = roc_auc_score(datatrain[self.ylabel],datatrain[probability]) 
            ax5 = fig.add_subplot(4,2,5)
            ax5.plot(tpr,fpr,color='blue',label='train_AUC=%.3f'%train_AUC) 
            ax5.plot([0,1],[0,1],'r--')
            ax5.set_ylim(0,1)
            ax5.set_xlim(0,1)
            ax5.set_title('ROC')
            ax5.legend(loc='best')

            tpr,fpr,threshold = roc_curve(datatest[self.ylabel],datatest[probability]) 
            test_AUC = roc_auc_score(datatest[self.ylabel],datatest[probability]) 
            ax6 = fig.add_subplot(4,2,6)
            ax6.plot(tpr,fpr,color='blue',label='test_AUC=%.3f'%test_AUC) 
            ax6.plot([0,1],[0,1],'r--')
            ax6.set_ylim(0,1)
            ax6.set_xlim(0,1)
            ax6.set_title('ROC')
            ax6.legend(loc='best')


            # 绘制训练集ks
            (train_ks_v,train_ks_df) = self.distin_ks(datatrain[self.ylabel],datatrain[probability])
            (test_ks_v,test_ks_df) = self.distin_ks(datatest[self.ylabel],datatest[probability])
            
            
            ax7 = fig.add_subplot(4,2,7)
            ax7.plot(train_ks_df['cum_pnt_all'].to_list(),train_ks_df['cum_pnt_bad'].to_list(),label='cum_pnt_bad')
            ax7.plot(train_ks_df['cum_pnt_all'].to_list(),train_ks_df['cum_pnt_good'].to_list(),label='cum_pnt_good')
            ax7.plot(train_ks_df['cum_pnt_all'].to_list(),train_ks_df['diff'].to_list(),label='diff')
            ax7.set(title='ks=%s'%round(train_ks_v,4),xlabel='cum_pnt_all',ylabel='cum_pnt')
            ax7.legend(fontsize=7)

            ax8 = fig.add_subplot(4,2,8)
            ax8.plot(test_ks_df['cum_pnt_all'].to_list(),test_ks_df['cum_pnt_bad'].to_list(),label='cum_pnt_bad')
            ax8.plot(test_ks_df['cum_pnt_all'].to_list(),test_ks_df['cum_pnt_good'].to_list(),label='cum_pnt_good')
            ax8.plot(test_ks_df['cum_pnt_all'].to_list(),test_ks_df['diff'].to_list(),label='diff')
            ax8.set(title='ks=%s'%round(test_ks_v,4),xlabel='cum_pnt_all',ylabel='cum_pnt')
            ax8.legend(fontsize=7)


            #sns.kdeplot(df_test[df_test['label'] == 0]['score_avg'],shade=True,color='r',label='Good Score Distribution')
            #sns.kdeplot(df_test[df_test['label'] == 1]['score_avg'],shade=True,color='b',label='Bad Score Distribution')
            plt.tight_layout()
            plt.savefig(imagePath+'/pic.png')

        except Exception as e:
            print('I_ModelDisplay,ModelDisplayClass,charge_plot',':',e) 

    def modelDisplayDF(self,datatrain,datatest,savePath,pre='score',preThreshold=0.05,probability='yPred'):
        """
        功能：输出整体的模型评价结果，包括两个excel，一个是分部及离线评价指标，一个是效果评估图像

        输入：
        datatrain:训练集数据集 dataframe
        datatest:测试集数据集 dataframe
        savePath:需要保存文档的路径
        pre:用户排序的列，经常性是score
        threshold:用于评测是否是坏样本的阈值
        probability: 预测的标记 string 

        输出：
        两个excel表

        管理记录：
        1、created by  mawenhui 2021/09/15
        '''
        """
        try:
            #计算分数分布
            trainEquiFrequent = self.group_by_list(datatrain,pre,'EquiFrequent')
            trainEquiDistance = self.group_by_list(datatrain,pre,'EquiDistance')
            testEquiFrequent = self.group_by_list(datatest,pre,'EquiFrequent')
            testEquiDistance =self.group_by_list(datatest,pre,'EquiDistance')           
            
            # 计算混淆矩阵
            (trainDf,trainRatioDf,traingainDf) = self.confusion_matrix(datatrain,probability,preThreshold)
            (testDf,testRatioDf,testgainDf) = self.confusion_matrix(datatest,probability,preThreshold)

            #计算模型的评估指标
            chargeDf = self.charge_matrix(datatrain,datatest,probability,preThreshold)

            # 绘制模型效果图
            self.charge_plot(datatrain,datatest,savePath,score=pre,probability=probability)
            # 保存结果数据
            with pd.ExcelWriter(os.path.join(savePath,'evaluate.xlsx')) as writer:
                trainEquiFrequent.to_excel(writer, sheet_name=u'评分卡分数分布-等频-训练集合',index=False)
                trainEquiDistance.to_excel(writer, sheet_name=u'评分卡分数分布-等距-训练集合',index=False)
                testEquiFrequent.to_excel(writer, sheet_name=u'评分卡分数分布-等频-测试集合',index=False)
                testEquiDistance.to_excel(writer, sheet_name=u'评分卡分数分布-等距-测试集合',index=False)
                chargeDf.to_excel(writer,sheet_name=u'模型效果评估指标')
            writer.save()

            workbook = xlsxwriter.Workbook(os.path.join(savePath,'evaluate_pic.xlsx'))
            worksheet1 = workbook.add_worksheet("评分卡运用效果")
            worksheet1.set_column('A:V', 20)
            worksheet1.write('A1', "训练集混淆矩阵")
            worksheet1.write('I1', "测试集混淆矩阵")
            worksheet1.write_row('A2',['','好用户','坏用户','总体'])
            worksheet1.add_table('A3:D5', {'data':np.array(trainDf.reset_index()).tolist(),'header_row': False})
            worksheet1.write_row('I2',['','好用户','坏用户','总体'])
            worksheet1.add_table('I3:L5', {'data':np.array(testDf.reset_index()).tolist(),'header_row': False})

            worksheet1.write('A7', "训练集混淆矩阵占比")
            worksheet1.write('I7', "测试集混淆矩阵占比")
            worksheet1.write_row('A8',['','好用户','坏用户','总体'])
            worksheet1.add_table('A9:D11', {'data':np.array(trainRatioDf.reset_index()).tolist(),'header_row': False})
            worksheet1.write_row('I8',['','好用户','坏用户','总体'])
            worksheet1.add_table('I9:L11', {'data':np.array(testRatioDf.reset_index()).tolist(),'header_row': False})

            worksheet1.write('A13', "训练集效果提升")
            worksheet1.write('I13', "测试集效果提升")
            worksheet1.write_row('A14',['','好用户','坏用户','总体'])
            worksheet1.add_table('A15:D16', {'data':np.array(traingainDf.reset_index()).tolist(),'header_row': False})
            worksheet1.write_row('I14',['','好用户','坏用户','坏样本占比'])
            worksheet1.add_table('I15:L16', {'data':np.array(testgainDf.reset_index()).tolist(),'header_row': False})

            worksheet2 = workbook.add_worksheet("模型效果评估图像")
            worksheet2.insert_image('B2',os.path.join(savePath,'pic.png'))
            workbook.close()

        except Exception as e:
            print('I_ModelDisplay,ModelDisplayClass,modelDisplayDF',':',e)