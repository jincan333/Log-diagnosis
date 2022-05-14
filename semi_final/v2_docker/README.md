一、算法逻辑
1. 分别读取日志数据和标签数据，并将其拼接成一份数据。
2. 对拼接后的数据，按照 sn,fault_time 两个维度进行划分，并将同一个sn两次fault_time之间的日志匹配到较大的fault_time。
3. 计算一些统计特征。
4. 读取词汇列表，计算词频特征。
5. 将所有特征和标签拼接。
6. 训练xgb模型。


二、代码运行说明
1. 直接运行run.sh文件即可输出预测结果。
2. 特征计算代码在/feature/xgb_v2_feature.py，模型训练代码在/model/xgb_model_v2.py，预测代码在/code/main.py。
3. 词汇列表在/user_data/words，特征计算结果在/user_data/feature_data，模型文件在/user_data/model_data，日志文件在/user_data/logs。


三、注意
1. 暂无