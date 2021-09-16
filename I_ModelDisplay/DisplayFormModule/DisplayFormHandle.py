import pandas as pd
import numpy as np

class DisplayFormClass():

    """
    功能描述：存储表格工具

    输入：
    path:  文件储存地址 
    sheet_name: 
    value1:
    workbook:
    style:


    输出：
    write_excel_xlsx: 表格储存工具


    管理记录：
    1. edited by 王文丹 2021/9/10
    """


    def write_excel_xlsx(path, sheet_name, value1,workbook, style = None):
        """
        功能描述：存储表格工具

        输入：
        path:  文件储存地址 
        sheet_name: 
        value1:
        workbook:
        style:


        输出：
        write_excel_xlsx: 表格储存工具


        管理记录：
        1. edited by 王文丹 2021/9/10
        """
        sheet = workbook.active
        sheet=workbook.create_sheet(title=sheet_name,index=0)
        m = len(value1)
        n = 2
        for c in range(m):
            if c == 0:
                n = 2
            else:
                
                n = n + len(value1[list(value1)[c-1]]) + 3
            value2 = value1[list(value1)[c]]
            columns1 = list(value2.columns)
            index1 = list(value2.index)
            for i in range(value2.shape[0]):
                for j in range(value2.shape[1]):
                    if style == None:
                        if i == 1 & i == 0:
                            sheet.cell(row=i+n,column=j+2,value=str(value2.iloc[i,j]))
                        else:
                            
                            sheet.cell(row=i+n,column=j+2,value="{0:.6f}".format(value2.iloc[i,j]))
                    else:
                        if j == 1 & j == 0:
                            sheet.cell(row=i+n,column=j+2,value=str(value2.iloc[i,j]))
                        else:
                            
                            sheet.cell(row=i+n,column=j+2,value="{0:.6f}".format(value2.iloc[i,j]))
            for j in range(len(columns1)):
                sheet.cell(row=n-1,column=j+2,value=str(columns1[j]))

            for i in range(len(index1)):
                sheet.cell(row=i+n,column=1,value=str(index1[i]))        
        workbook.save(path)
        workbook.close()
        print(sheet_name+"xlsx格式表格写入数据成功！")