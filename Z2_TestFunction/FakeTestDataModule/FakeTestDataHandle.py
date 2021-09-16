from faker import Faker
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
fake = Faker('zh_CN')

class FakeTestDataClass():

    """
    功能描述：利用faker库造数，编写造数小脚本
    
    输入：
    numlength: 造数的数据长度
    missrate: 数据缺失率list 如缺失率[0,0,1,0,2,0.3]
    leftDigits: 造数据小数点左边数据的位数
    rightDigits: 造数据小数点右边数据的位数
    positive：是否要求正值 True 或者 False
    min_value: 最小值
    max_value: 最大值
    scatterList：离散列表，根据该列表生成离散型数据

    输出：
    函数fake_float_and_nan: 利用faker造数值型数据，可自由选择数据缺失率
    函数fake_scatter_and_nan: 利用faker造离散型数据，可自由选择数据缺失率
    函数fake_certid: 利用faker造身份证号码数据
    函数fake_bad01：利用faker造好坏标签数据，支持填入坏样本率
    
    管理记录：
    1. edited by 王文丹 2021/07/24

    """

    def __init__(self, numlength, missrate):

        self.numlength = numlength # 造数的数据长度
        self.missrate = missrate # 数据缺失率list 如缺失率[0,0,1,0,2,0.3]

    def fake_float_and_nan(self,leftDigits, rightDigits, positive, minValue, maxValue): 
        """
        功能描述：利用faker造数值型数据，可自由选择数据缺失率

        输入：
        numlength：需要造数的数据长度
        missrate：数据缺失率
        leftDigits: 造数据小数点左边数据的位数
        rightDigits: 造数据小数点右边数据的位数
        positive：是否要求正值 True 或者 False
        min_value: 最小值
        max_value: 最大值

        输出：
        datafake: 造出的单列pandas的数值型数据

        管理记录：
        1. edited by 王文丹 2021/07/24
        """

        fake = Faker('zh_CN')
        numlengthIsmiss = round(self.numlength*self.missrate) # 计算非空的造数长度
        numlengthNotmiss = self.numlength - numlengthIsmiss # 计算非空的造数长度
        datafake = []
        for i in range(numlengthNotmiss):
            datafake.append(fake.pydecimal(leftDigits, rightDigits, positive, minValue, maxValue))
        datafake = pd.concat([pd.DataFrame(datafake),pd.DataFrame(np.repeat(np.nan,numlengthIsmiss,axis=0))])
        datafake = shuffle(datafake) # 数据打乱

        return datafake
    
    def fake_scatter_and_nan(self,scatterList):

        """
        功能描述：利用faker造离散型数据，可自由选择数据缺失率

        输入：
        numlength：需要造数的数据长度
        missrate：数据缺失率
        scatterList：离散列表，根据该列表生成离散型数据

        输出：
        datafake: 造出的单列pandas的数值型数据

        管理记录：
        1. edited by 王文丹 2021/07/24
        """
        numlengthIsmiss = round(self.numlength*self.missrate) # 计算非空的造数长度
        numlengthNotmiss = self.numlength - numlengthIsmiss # 计算非空的造数长度
        datafake = []
        for i in range(numlengthNotmiss):
            datafake.append(fake.random_element(elements=scatterList))
        datafake = pd.concat([pd.DataFrame(datafake),pd.DataFrame(np.repeat(np.nan,numlengthIsmiss,axis=0))])
        datafake = shuffle(datafake) # 数据打乱 

        return datafake 
    
    def fake_certid(self,numlength):
        """
        功能描述：利用faker造身份证号码数据
        
        输入：
        numlength：需要造数的数据长度
        
        输出：
        datafake: 造出的单列pandas的数值型数据
        
        管理记录：
        1. edited by 王文丹 2021/07/24  
        """
        datafake = []
        for i in range(numlength):
            datafake.append(fake.ssn())
        return datafake 
    
    def fake_bad01(self,badrate):

        """
        功能描述：利用faker造好坏标签数据，支持填入坏样本率
        
        输入：
        numlength：需要造数的数据长度
        missrate：数据缺失率
        badrate: 坏样本率
        
        输出：
        datafake: 造出的单列pandas的0-1数据

        管理记录：
        1. edited by 王文丹 2021/07/24  
        """
        numlength1 = round(self.numlength*badrate)    # 计算非空的造数长度0
        numlength0 = self.numlength - numlength1      # 计算非空的造数长度1
        datafake1 = pd.DataFrame(np.repeat(1,numlength1,axis=0))
        datafake0 = pd.DataFrame(np.repeat(0,numlength0,axis=0))                        
        datafake = pd.concat([datafake1,datafake0],axis=0)
        datafake = shuffle(datafake) # 数据打乱
        return datafake



