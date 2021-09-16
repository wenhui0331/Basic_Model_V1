# 开发构建流程

## 一、通过命令行运行cd，切换到项目根目录下。
## 二、根据需求设计文档，进入对应模块进行开发。
## 三、在J_unintes下编写对应模块的测试文件。
## 四、运行 python -m unitest /测试模块对应的路径 进行测试


# 建模代码平台

## 一、A_GetData：数据获取包
### 1.DocumentsInputModule/DocumentsInputHandle/DocumentsInputClass/document_input： 网络直连函数

### 2. InternetInoutModule: 网络直连模块

## 二、B_ExploreData：数据探索性分析

## 三、C_PreprocessData：数据预处理包
### 1. DictValueModule/DictValueHandle/dict_transform: 字典转换函数
### 2. MissValueModule/MissValueHandle/output_var_type：变量描述性概况函数
### 3. MissValueModule/MissValueHandle/missing_cal：变量描述性概况函数（加缺失情况）
### 4. MissValueModule/MissValueHandle/missing_delete_var：缺失率高于阀值，将该变量删除
### 5. MissValueModule/MissValueHandle/missing_fill_var：缺失值填充函数
### 6. RepeatValueModule/RepeatValueHandle/repeat_drop_value: 删除重复值、空行
### 7. RepeatValueModule/RepeatValueHandle/single_drop_value: 单一值处理函数
### 8. TransformValueModule/TransformValueHandle/transform_object_dict: 依照坏样本率排序形成映射字典
### 9. TransformValueModule/TransformValueHandle/transform_object_type: 字符型变量转换为数值型变量
### 10. ZPreprocessCallModule/ZPreprocessCallHandle/Preprocess_all_Call: 数据清洗包的总调用函数
### 11. ZPreprocessCallModule/ZPreprocessCallHandle/ParaConfig.json: json文件


## 四、D_SplitData：数据分箱

## 五、E_BinResult：分箱结果运用

## 六、F_SelectChar：变量筛选

## 七、G_ModelScore：评分卡分数输出

## 八、H_ModelEvaluate：评分卡效果评估指标

## 九、I_ModelDisplay：评分卡结果输出展示

## 十：J_unintest：单元测试模块


