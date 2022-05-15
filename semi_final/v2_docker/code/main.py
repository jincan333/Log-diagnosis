# -*- coding:utf-8 -*-

import sys
import os
import pandas as pd
import warnings
import logging
from logging import handlers
from datetime import datetime, date
import numpy as np
import xgboost as xgb
import traceback
import re
import random

## 日志格式设置
# 日志级别关系映射
level_relations = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'crit': logging.CRITICAL
}


def get_logger(filename, level='info'):
    # 创建日志对象
    log = logging.getLogger(filename)
    # 设置日志级别
    log.setLevel(level_relations.get(level))
    # 日志输出格式
    fmt = logging.Formatter('%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    # 输出到控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    # 输出到文件
    # 日志文件按天进行保存，每天一个日志文件
    file_handler = handlers.TimedRotatingFileHandler(filename=filename, when='D', backupCount=1, encoding='utf-8')
    # 按照大小自动分割日志文件，一旦达到指定的大小重新生成文件
    # file_handler = handlers.RotatingFileHandler(filename=filename, maxBytes=1*1024*1024*1024, backupCount=1, encoding='utf-8')
    file_handler.setFormatter(fmt)
    if not log.handlers:
        log.addHandler(console_handler)
        log.addHandler(file_handler)
    return log


# sn分组后，本次报错和上次报错之间的日志匹配到本次报错
def divideLogByFaultTime(log_label_df: pd.DataFrame):
    log_correspond_label_df = pd.DataFrame(columns=['sn', 'fault_time', 'msg', 'time', 'server_model', 'label'])
    no_label_log_list = []
    log_label_df = log_label_df.reset_index(drop=True)

    for sn, log in log_label_df.groupby('sn'):
        if len(log[log['label'] != '']) == 0:
            no_label_log_list.append(log)
        elif len(log[log['label'] != '']) == 1:
            msg_df = log[log['label'] == '']
            msg_df['label'] = log[log['label'] != '']['label'].iloc[0]
            msg_df['fault_time'] = log[log['label'] != '']['time'].iloc[0]
            log_correspond_label_df = pd.concat([log_correspond_label_df, msg_df])
        else:
            # 使用index的顺序取数时，要注意index必须按所需的顺序排列
            cutoff_index = [-1] + log.loc[log['label'] != ''].index.tolist() + [log.index.tolist()[-1] + 1]
            for kth in range(len(cutoff_index) - 1):
                temp_log = log.loc[(log.index <= cutoff_index[kth + 1]) & (log.index > cutoff_index[kth])]
                if len(temp_log) > 0:
                    if len(temp_log[temp_log['label'] != '']) == 0:
                        no_label_log_list.append(temp_log)
                    # 只有标签，没有日志的数据，把标签的部分数据直接作为日志
                    elif len(temp_log) == 1:
                        msg_df = temp_log
                        msg_df['fault_time'] = temp_log[temp_log['label'] != '']['time'].iloc[0]
                        log_correspond_label_df = pd.concat([log_correspond_label_df, msg_df])
                    else:
                        msg_df = temp_log[temp_log['label'] == '']
                        msg_df['label'] = temp_log[temp_log['label'] != '']['label'].iloc[0]
                        msg_df['fault_time'] = temp_log[temp_log['label'] != '']['time'].iloc[0]
                        log_correspond_label_df = pd.concat([log_correspond_label_df, msg_df])
    return log_correspond_label_df, no_label_log_list


# sn分组后，按照最近邻+时间间隔划分日志数据
def divideLogByNearestTime(log_label_df: pd.DataFrame):
    log_correspond_label_df = pd.DataFrame(columns=['sn', 'fault_time', 'msg', 'time', 'server_model', 'label'])
    origin_label_df = log_label_df[log_label_df['fault_time'] != '']
    no_label_log_list = []
    cutoff = 10 * 3600

    for sn, log in log_label_df.groupby('sn'):
        if len(log[log['label'] != '']) == 0:
            no_label_log_list.append(log)
        elif len(log[log['label'] != '']) == 1:
            msg_df = log[log['label'] == '']
            if len(msg_df) > 0:
                msg_df['label'] = log[log['label'] != '']['label'].iloc[0]
                msg_df['fault_time'] = log[log['label'] != '']['time'].iloc[0]
                log_correspond_label_df = pd.concat([log_correspond_label_df, msg_df])
        else:
            lable_df = log[log['label'] != '']
            msg_df = log[log['label'] == '']
            for msg_item in msg_df.iterrows():
                previous_delta_time = 1000 * 24 * 3600
                for lable_item in lable_df.iterrows():
                    now_delta_time = abs(datetime.strptime(lable_item[1]['time'], '%Y-%m-%d %H:%M:%S'
                                                           ) - datetime.strptime(msg_item[1]['time'],
                                                                                 '%Y-%m-%d %H:%M:%S'))
                    if now_delta_time.days * 24 * 3600 + now_delta_time.seconds < previous_delta_time:
                        previous_delta_time = now_delta_time.days * 24 * 3600 + now_delta_time.seconds
                        final_lable = lable_item[1]
                        if previous_delta_time < cutoff:
                            msg_item[1]['fault_time'] = lable_item[1]['time']
                            msg_item[1]['label'] = lable_item[1]['label']
            log_correspond_label_df = pd.concat([log_correspond_label_df, msg_df])
    log_correspond_label_df = log_correspond_label_df[log_correspond_label_df['label'] != '']
    # 找出没有匹配到日志的标签并将其添加到 日志标签映射表 中
    temp_df = pd.concat([log_correspond_label_df, origin_label_df])
    sn_list = []
    fault_time_list = []
    msg_list = []
    time_list = []
    server_model_list = []
    label_list = []
    for g in temp_df.groupby(['sn', 'fault_time', 'label']):
        if len(g[1]) == 1:
            sn_list.append(g[0][0])
            fault_time_list.append(g[0][1])
            msg_list.append('')
            time_list.append(g[1]['time'].iloc[0])
            server_model_list.append(g[1]['server_model'].iloc[0])
            label_list.append(g[0][2])
    no_log_label_df = pd.DataFrame({
        'sn': sn_list,
        'fault_time': fault_time_list,
        'msg': msg_list,
        'time': time_list,
        'server_model': server_model_list,
        'label': label_list
    })
    log_correspond_label_df = pd.concat([log_correspond_label_df, no_log_label_df])
    return log_correspond_label_df, no_label_log_list


# 计算统计特征
def calculateStatisticFeature(log_correspond_label_df: pd.DataFrame) -> pd.DataFrame:
    use_log_label_df = log_correspond_label_df

    use_log_label_df['msg_hour'] = use_log_label_df['time'].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)
    use_log_label_df['msg_minute'] = use_log_label_df['time'].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").minute)
    use_log_label_df['fault_hour'] = use_log_label_df['fault_time'].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)
    use_log_label_df['fault_minute'] = use_log_label_df['fault_time'].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").minute)

    # 0408新增
    # 最近一次日志时间距报错时间间隔，单位秒
    nearest_msg_fault_time_delta_list = []
    # 日志不去重时长度1,2,3,4日志数量统计
    all_msg_1_cnt_list = []
    all_msg_2_cnt_list = []
    all_msg_3_cnt_list = []
    all_msg_4_cnt_list = []

    fault_minute_list = []
    msg_1_cnt_list = []
    msg_2_cnt_list = []
    msg_3_cnt_list = []
    msg_4_cnt_list = []
    msg_hour_max_list = []
    msg_hour_min_list = []
    msg_hour_avg_list = []
    msg_hour_median_list = []
    msg_hour_mode_list = []
    msg_minute_max_list = []
    msg_minute_min_list = []
    msg_minute_avg_list = []
    msg_minute_median_list = []
    msg_minute_mode_list = []

    sn_list = []
    server_model_list = []
    msg_log_list = []
    msg_cnt_list = []
    fault_hour_list = []
    label_list = []
    fault_time_list = []
    for msg_log_df in use_log_label_df.groupby(['sn', 'fault_time', 'label']):
        msg_log_str = ''
        all_msg_1_cnt = 0
        all_msg_2_cnt = 0
        all_msg_3_cnt = 0
        all_msg_4_cnt = 0
        msg_1_cnt = 0
        msg_2_cnt = 0
        msg_3_cnt = 0
        msg_4_cnt = 0
        for info in msg_log_df[1]['msg']:
            if info == info:
                if len(info.split('|')) == 1:
                    all_msg_1_cnt += 1
                elif len(info.split('|')) == 2:
                    all_msg_2_cnt += 1
                elif len(info.split('|')) == 3:
                    all_msg_3_cnt += 1
                else:
                    all_msg_4_cnt += 1
        for info in msg_log_df[1]['msg'].drop_duplicates():
            if info == info:
                msg_log_str = msg_log_str + info.lower() + '.'
                if len(info.split('|')) == 1:
                    msg_1_cnt += 1
                elif len(info.split('|')) == 2:
                    msg_2_cnt += 1
                elif len(info.split('|')) == 3:
                    msg_3_cnt += 1
                else:
                    msg_4_cnt += 1
        nearest_msg_fault_time_delta = abs(datetime.strptime(msg_log_df[1].iloc[-1]['time'], '%Y-%m-%d %H:%M:%S'
                                                             ) - datetime.strptime(msg_log_df[0][1],
                                                                                   '%Y-%m-%d %H:%M:%S'))
        nearest_msg_fault_time_delta = nearest_msg_fault_time_delta.days * 24 * 3600 + nearest_msg_fault_time_delta.seconds
        sm = int(msg_log_df[1].iloc[0]['server_model'][2:])

        sn_list.append(msg_log_df[0][0])
        fault_time_list.append(msg_log_df[0][1])
        label_list.append(msg_log_df[0][2])

        nearest_msg_fault_time_delta_list.append(nearest_msg_fault_time_delta)
        server_model_list.append(sm)
        msg_log_list.append(msg_log_str)
        msg_cnt_list.append(len(msg_log_df[1]))

        fault_hour_list.append(msg_log_df[1].iloc[0]['fault_hour'])
        fault_minute_list.append(msg_log_df[1].iloc[0]['fault_minute'])

        all_msg_1_cnt_list.append(all_msg_1_cnt)
        all_msg_2_cnt_list.append(all_msg_2_cnt)
        all_msg_3_cnt_list.append(all_msg_3_cnt)
        all_msg_4_cnt_list.append(all_msg_4_cnt)

        msg_1_cnt_list.append(msg_1_cnt)
        msg_2_cnt_list.append(msg_2_cnt)
        msg_3_cnt_list.append(msg_3_cnt)
        msg_4_cnt_list.append(msg_4_cnt)

        msg_hour_max_list.append(msg_log_df[1]['msg_hour'].max())
        msg_hour_min_list.append(msg_log_df[1]['msg_hour'].min())
        msg_hour_avg_list.append(msg_log_df[1]['msg_hour'].mean())
        msg_hour_median_list.append(msg_log_df[1]['msg_hour'].median())
        msg_hour_mode_list.append(msg_log_df[1]['msg_hour'].mode()[0])

        msg_minute_max_list.append(msg_log_df[1]['msg_minute'].max())
        msg_minute_min_list.append(msg_log_df[1]['msg_minute'].min())
        msg_minute_avg_list.append(msg_log_df[1]['msg_minute'].mean())
        msg_minute_median_list.append(msg_log_df[1]['msg_minute'].median())
        msg_minute_mode_list.append(msg_log_df[1]['msg_minute'].mode()[0])

    msg_log_label_df = pd.DataFrame(
        {
            'sn': sn_list,
            'fault_time': fault_time_list,
            'server_model': server_model_list,
            'msg_cnt': msg_cnt_list,
            'fault_hour': fault_hour_list,
            'fault_minute': fault_minute_list,
            'nearest_msg_fault_time_delta': nearest_msg_fault_time_delta_list,
            'all_msg_1_cnt': all_msg_1_cnt_list,
            'all_msg_2_cnt': all_msg_2_cnt_list,
            'all_msg_3_cnt': all_msg_3_cnt_list,
            'all_msg_4_cnt': all_msg_4_cnt_list,
            'msg_1_cnt': msg_1_cnt_list,
            'msg_2_cnt': msg_2_cnt_list,
            'msg_3_cnt': msg_3_cnt_list,
            'msg_4_cnt': msg_4_cnt_list,
            'msg_hour_max': msg_hour_max_list,
            'msg_hour_min': msg_hour_min_list,
            'msg_hour_avg': msg_hour_avg_list,
            'msg_hour_median': msg_hour_median_list,
            'msg_hour_mode': msg_hour_mode_list,
            'msg_minute_max': msg_minute_max_list,
            'msg_minute_min': msg_minute_min_list,
            'msg_minute_avg': msg_minute_avg_list,
            'msg_minute_median': msg_minute_median_list,
            'msg_minute_mode': msg_minute_mode_list,
            'msg_log': msg_log_list,
            'label': label_list
        }
    )
    return msg_log_label_df


# 计算特征函数
def caculateFeature(log_df: pd.DataFrame, label_df: pd.DataFrame, word_list: list) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    logger = get_logger('./user_data/logs/{}_info.log'.format(date.today()), 'info')
    logger_error = get_logger('./user_data/logs/{}_error.log'.format(date.today()), 'error')

    logger.info('开始拼接日志和标签数据')
    log_df['label'] = ''
    log_df['fault_time'] = ''
    log_df = log_df[['sn', 'fault_time', 'msg', 'time', 'server_model', 'label']]

    label_df['time'] = label_df['fault_time']
    label_df['msg'] = ''
    label_df['server_model'] = label_df['sn'].map(dict(zip(log_df['sn'], log_df['server_model'])))
    label_df = label_df[['sn', 'fault_time', 'msg', 'time', 'server_model', 'label']]
    log_label_df = pd.concat([log_df, label_df], axis=0).sort_values(by='time')
    #     log_label_df['fault_time'] = ''
    log_label_df = log_label_df[['sn', 'fault_time', 'msg', 'time', 'server_model', 'label']]
    logger.info('拼接日志和标签数据结束')

    logger.info('开始匹配日志和标签')
    logger.info('使用报错时间截断进行划分')
    # 使用报错时间截断进行划分
    #     NearestTime_log_correspond_label_df, NearestTime_no_label_log_list = divideLogByNearestTime(log_label_df)
    #     NearestTime_log_correspond_label_df.to_csv('./user_data/tmp_data/NearestTime_log_correspond_label_df.csv', index = None)
    NearestTime_log_correspond_label_df = pd.read_csv('./user_data/tmp_data/NearestTime_log_correspond_label_df.csv')
    #     FaultTime_log_correspond_label_df, FaultTime_no_label_log_list = divideLogByFaultTime(log_label_df)
    #     FaultTime_log_correspond_label_df.to_csv('./user_data/tmp_data/FaultTime_log_correspond_label_df.csv', index = None)
    FaultTime_log_correspond_label_df = pd.read_csv('./user_data/tmp_data/FaultTime_log_correspond_label_df.csv')

    NearestTime_log_correspond_label_df = NearestTime_log_correspond_label_df[
        NearestTime_log_correspond_label_df['msg'].notna()]
    FaultTime_log_correspond_label_df = FaultTime_log_correspond_label_df[
        FaultTime_log_correspond_label_df['msg'].notna()]
    logger.info('匹配日志和标签结束')

    logger.info('开始计算统计特征')
    # 使用报错时间截断进行划分
    msg_log_label_df = calculateStatisticFeature(NearestTime_log_correspond_label_df)
    logger.info('计算统计特征结束')

    msg_log_list = list(msg_log_label_df['msg_log'])
    label_list = list(msg_log_label_df['label'])

    # 计算词频向量
    logger.info('开始计算词频特征')
    frequency_vector_list = []
    tag = 0
    for word in word_list:
        if tag % 100 == 0:
            print(tag, datetime.now())
        pattern = re.compile(word)
        frequency_vector = [len(re.findall(pattern, log)) for log in msg_log_list]
        frequency_vector_list.append(frequency_vector)
        tag += 1
    logger.info('计算词频特征结束')

    frequency_vector_df = pd.DataFrame(frequency_vector_list)
    frequency_vector_df = frequency_vector_df.T
    frequency_vector_df.columns = word_list
    statistic_feature_list = list(msg_log_label_df.columns)[2:-2]
    feature_df = frequency_vector_df
    feature_df[statistic_feature_list] = msg_log_label_df[statistic_feature_list]

    feature_df['label'] = label_list
    feature_df[['sn', 'fault_time']] = msg_log_label_df[['sn', 'fault_time']]
    logger.info('最后3列为: label, sn, fault_time, 其余列均为特征')
    logger.info('数据条数: {}, 特征个数: {}'.format(feature_df.shape[0], feature_df.shape[1] - 3))
    return feature_df


# xgb模型参数
xgb_params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类问题
    'num_class': 4,  # 类别数，与multi softmax并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数，越大越保守，一般0.1 0.2的样子
    'max_depth': 6,  # 构建树的深度，越大越容易过拟合
    'lambda': 2,  # 控制模型复杂度的权重值的L2 正则化项参数，参数越大，模型越不容易过拟合
    'subsample': 1,  # 随机采样训练样本
    'colsample_bytree': 1,  # 这个参数默认为1，是每个叶子里面h的和至少是多少
    # 对于正负样本不均衡时的0-1分类而言，假设h在0.01附近，min_child_weight为1
    # 意味着叶子节点中最少需要包含100个样本。这个参数非常影响结果，
    # 控制叶子节点中二阶导的和的最小值，该参数值越小，越容易过拟合
    'silent': 0,  # 设置成1 则没有运行信息输入，最好是设置成0
    'eta': 0.3,  # 如同学习率
    'seed': 1000,
    'nthread': 16,  # CPU线程数
    # 'eval_metric':'auc'
}


# 指标评估
def macro_f1(label_df: pd.DataFrame, prediction_df: pd.DataFrame) -> float:
    """
    计算得分
    :param label_df: [sn,fault_time,label]
    :param prediction_df: [sn,fault_time,label]
    :return:
    """
    warnings.filterwarnings("ignore")
    logger = get_logger('./user_data/logs/{}_info.log'.format(date.today()), 'info')
    logger_error = get_logger('./user_data/logs/{}_error.log'.format(date.today()), 'error')

    prediction_df.columns = ['sn', 'fault_time', 'prediction']
    outcome_df = pd.merge(label_df, prediction_df, how='left', on=['sn', 'fault_time'])
    weights = [5 / 11, 4 / 11, 1 / 11, 1 / 11]
    macro_F1 = 0.
    for i in range(len(weights)):
        TP = len(outcome_df[(outcome_df['label'] == i) & (outcome_df['prediction'] == i)])
        FP = len(outcome_df[(outcome_df['label'] != i) & (outcome_df['prediction'] == i)])
        FN = len(outcome_df[(outcome_df['label'] == i) & (outcome_df['prediction'] != i)])
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        macro_F1 += weights[i] * F1
        logger.info('Label {}:   Precision {: .2f}, Recall {: .2f}, F1 {: .2f}'.format(i, precision, recall, F1))
    logger.info('macro_f1: {}\n'.format(macro_F1))

    return macro_F1


# 模型训练函数
def xgbTrain(feature_df: pd.DataFrame) -> xgb.XGBModel:
    '''
    feature_df: 要求最后3列为: label, sn, fault_time, 其余列均为特征
    '''
    warnings.filterwarnings("ignore")
    logger = get_logger('./user_data/logs/{}_info.log'.format(date.today()), 'info')
    logger_error = get_logger('./user_data/logs/{}_error.log'.format(date.today()), 'error')

    feature_name_list = list(feature_df.columns)[0:-3]
    feature = np.array(feature_df[feature_name_list])
    label = np.array(feature_df['label'])
    label_df = feature_df[['sn', 'fault_time', 'label']]
    prediction_df = feature_df[['sn', 'fault_time']]

    train_data = xgb.DMatrix(feature, label=label)
    train_feature = xgb.DMatrix(feature)
    logger.info('开始训练xgb模型')
    xgb_model = xgb.train(xgb_params, train_data, num_boost_round=500)
    logger.info('训练xgb模型结束')
    # 训练集指标评估
    prediction = xgb_model.predict(train_feature)
    prediction_df['label'] = prediction
    logger.info('训练集评估效果: ')
    macro_f1(label_df, prediction_df)

    return xgb_model


# xgb模型预测函数
def xgbPredict(model: xgb.XGBModel, feature_df: pd.DataFrame, label_df=None) -> pd.DataFrame:
    '''
        feature_df: 要求最后3列为: label, sn, fault_time, 其余列均为特征
    '''
    warnings.filterwarnings("ignore")
    logger = get_logger('./user_data/logs/{}_info.log'.format(date.today()), 'info')
    logger_error = get_logger('./user_data/logs/{}_error.log'.format(date.today()), 'error')

    if label_df is None:
        feature_name_list = list(feature_df.columns)[0:-3]
        feature = np.array(feature_df[feature_name_list])
        prediction_df = feature_df[['sn', 'fault_time']]

        test_feature = xgb.DMatrix(feature)
        logger.info('开始xgb模型预测')
        prediction = model.predict(test_feature)
        logger.info('xgb模型预测结束')
        prediction_df['label'] = prediction
        prediction_df['label'] = prediction_df['label'].apply(lambda x: int(x))

    else:
        feature_name_list = list(feature_df.columns)[0:-3]
        feature = np.array(feature_df[feature_name_list])
        prediction_df = feature_df[['sn', 'fault_time']]

        test_feature = xgb.DMatrix(feature)
        logger.info('开始xgb模型预测')
        prediction = model.predict(test_feature)
        logger.info('xgb模型预测结束')
        # 测试集指标评估
        prediction_df['label'] = prediction
        prediction_df['label'] = prediction_df['label'].apply(lambda x: int(x))
        logger.info('测试集评估效果: ')
        macro_f1(label_df, prediction_df)

    return prediction_df


# xgb模型随机训练并投票预测
def xgbRandomTrainPredict(train_feature_df: pd.DataFrame, test_feature_df: pd.DataFrame, label_df=None) -> pd.DataFrame:
    ## 每个子模型样本均衡，利用投票规则生成最终预测
    random.seed(0)
    N = 50  # number of the models
    num_sample = 700  # number of samples for each label

    _label0_index_list = list(train_feature_df[train_feature_df['label'] == 0].index)
    _label1_index_list = list(train_feature_df[train_feature_df['label'] == 1].index)
    _label2_index_list = list(train_feature_df[train_feature_df['label'] == 2].index)
    _label3_index_list = list(train_feature_df[train_feature_df['label'] == 3].index)
    feature_name_list = list(train_feature_df.columns)[0:-3]
    test_feature = np.array(test_feature_df[feature_name_list])
    test_feature = xgb.DMatrix(test_feature)
    prediction_df = test_feature_df[['sn', 'fault_time']]

    for iter in np.arange(N):
        idx_0 = random.sample(_label0_index_list, num_sample)
        idx_1 = random.sample(_label1_index_list, num_sample)
        idx_2 = random.sample(_label2_index_list, num_sample)
        idx_3 = random.sample(_label3_index_list, num_sample)
        idx = np.hstack((idx_0, idx_1, idx_2, idx_3))
        random.shuffle(idx)
        sub_train_feature_df = train_feature_df.loc[idx, :]
        sub_train_feature = np.array(sub_train_feature_df[feature_name_list])
        sub_train_label = np.array(sub_train_feature_df['label'])
        sub_train_data = xgb.DMatrix(sub_train_feature, label=sub_train_label)

        logger.info('开始第{}轮训练和预测'.format(iter))
        sub_xgb_model = xgb.train(xgb_params, sub_train_data, num_boost_round=500)
        sub_test_pred = sub_xgb_model.predict(test_feature)
        if iter == 0:
            val_pred = sub_test_pred
        else:
            val_pred = np.vstack((val_pred, sub_test_pred))
        logger.info('第{}轮训练和预测结束'.format(iter))

    # 训练集指标评估
    final_pred = [np.argmax(np.bincount(val_pred[:, i].astype(int))) for i in np.arange(val_pred.shape[1])]
    final_pred = np.array(final_pred).astype(int)
    prediction_df['label'] = final_pred
    logger.info('训练集评估效果: ')
    macro_f1(label_df, prediction_df)

    return prediction_df




if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.path[0]))
    # 忽略warning
    warnings.filterwarnings("ignore")
    logger = get_logger('./user_data/logs/{}_info.log'.format(date.today()), 'info')
    logger_error = get_logger('./user_data/logs/{}_error.log'.format(date.today()), 'error')

    # A榜
    # final_log_a = pd.read_csv('./tcdata/final_sel_log_dataset_a.csv')
    # final_label_a = pd.read_csv('./tcdata/final_submit_dataset_a.csv')
    # final_label_a['label'] = -1
    # important_word_list = list(pd.read_csv('./user_data/words/important_word_df.csv')['word'])
    # word_list = list(set(important_word_list))
    # final_a_feature_df = caculateFeature(final_log_a, final_label_a, word_list)
    # train_feature_df = pd.read_csv('./user_data/feature_data/train_feature_v2p1_df.csv')
    # final_a_feature_df = final_a_feature_df[list(train_feature_df)]
    # final_a_feature_df.to_csv('./user_data/feature_data/final_a_feature_df.csv', index=None)
    # model = xgb.Booster()
    # model.load_model('./user_data/model_data/xgb_model_v2p1.json')
    # final_prediction_df = xgbPredict(model, final_a_feature_df)
    # final_prediction_df.to_csv('./prediction_result/predictions.csv',index=None)

    # B榜
    final_log_b = pd.read_csv('./tcdata/final_sel_log_dataset_b.csv')
    final_label_b = pd.read_csv('./tcdata/final_submit_dataset_b.csv')
    final_label_b['label'] = -1
    important_word_list = list(pd.read_csv('./user_data/words/important_word_df.csv')['word'])
    word_list = list(set(important_word_list))
    final_b_feature_df = caculateFeature(final_log_b, final_label_b, word_list)
    train_feature_df = pd.read_csv('./user_data/feature_data/train_feature_v2p1_df.csv')
    final_b_feature_df = final_b_feature_df[list(train_feature_df)]
    final_b_feature_df.to_csv('./user_data/feature_data/final_b_feature_df.csv', index=None)
    model = xgb.Booster()
    model.load_model('./user_data/model_data/xgb_model_v2p1.json')
    final_prediction_df = xgbPredict(model, final_b_feature_df)
    final_prediction_df.to_csv('./prediction_result/predictions.csv',index=None)







