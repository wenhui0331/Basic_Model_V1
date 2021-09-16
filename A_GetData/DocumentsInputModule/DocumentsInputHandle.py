import pandas as pd

class DocumentsInputClass():

    """

    功能：文档导入函数，导入csv/excel/txt文档

    输入：
    dataInputPath: 数据所在的路径，如'D:/1.csv' str
    dataStyle: 'csv'/'txt'/'excel'  str

    输出：
    函数dataForInput: 读取完成的数据 Dataframe

    维护记录:
    1. edited by 王文丹 2021/7/3
    
    """

    def document_input(self,dataInputPath,dataStyle):

        try:

            if dataStyle == 'csv':

                dataForInput = pd.read_csv(dataInputPath)

            elif dataStyle == 'txt':

                dataForInput = pd.read_csv(dataInputPath,sep='/t')

            elif dataStyle == 'excel':

                dataForInput = pd.read_excel(dataInputPath)

            return dataForInput

        except Exception as e:

            print('A_GetData, DocumentsInputModule, DocumentsInputHandle, DocumentsInputClass, document_input',':',e)






