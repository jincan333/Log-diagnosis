# -*- coding:utf-8 -*-


import pandas as pd
import numpy as np
import xgboost as xgb
import random
import pickle
import traceback
from datetime import datetime, date
import logging
from logging import handlers
import sys
import os
import warnings


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
    outcome_df = pd.merge(label_df, prediction_df ,how = 'left', on = ['sn', 'fault_time'])
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
def xgbPredict(model: xgb.XGBModel, feature_df: pd.DataFrame, label_df = None) -> pd.DataFrame:
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


if __name__ == '__main__':
    # 更改工作目录为当前项目根目录
    os.chdir(os.path.dirname(sys.path[0]))
    # 忽略warning
    warnings.filterwarnings("ignore")
    logger = get_logger('./user_data/logs/{}_info.log'.format(date.today()), 'info')
    logger_error = get_logger('./user_data/logs/{}_error.log'.format(date.today()), 'error')

    feature_df = pd.read_csv('./user_data/feature_data/xgb_model_v2.csv')
    model = xgbTrain(feature_df)
    prediction_df = xgbPredict(model, feature_df, feature_df[['sn', 'fault_time', 'label']])

    # status = 1
    # # 读取特征数据
    # try:
    #     feature_df = pd.read_csv('./user_data/feature_data/xgb_model_v2.csv')
    #     feature_name_list = list(feature_df.columns)[0:-3]
    #     feature = np.array(feature_df[feature_name_list])
    #     label = np.array(feature_df['label'])
    # except:
    #     logger_error.error('读取特征计算结果出错, 报错详细信息: ', traceback.format_exc())
    #     status = 0
    #
    #
    # 训练验证集模型
    # if status == 1:
    #     try:
    #         random.seed(0)
    #         val_mask = [random.random() < 0.3 for _ in range(len(feature))]
    #         train_mask = [not xx for xx in val_mask]
    #         val_feature = feature[val_mask]
    #         val_label = label[val_mask]
    #         train_feature = feature[train_mask]
    #         train_label = label[train_mask]
    #         train_data=xgb.DMatrix(train_feature, label=train_label)
    #         train_feature=xgb.DMatrix(train_feature)
    #         val_feature=xgb.DMatrix(val_feature)
    #         logger.info('开始训练验证集模型')
    #         xgb_model=xgb.train(xgb_params,train_data,num_boost_round=500)
    #         logger.info('训练验证集模型结束')
    #         train_pred=xgb_model.predict(train_feature)
    #         val_pred=xgb_model.predict(val_feature)
    #         val_macro_f1 = macro_f1(val_label,val_pred)
    #         logger.info('验证集 macro_f1 为: {}'.format(val_macro_f1))
    #     except:
    #         logger_error.error('训练验证集模型出错, 报错详细信息: ', traceback.format_exc())
    #         status = 0
    #
    # # 训练测试集模型
    # if status == 1:
    #     try:
    #         all_data = xgb.DMatrix(feature, label = label)
    #
    #         logger.info('开始训练验证集模型')
    #         xgb_model_v2 = xgb.train(xgb_params, all_data, num_boost_round=500)
    #         logger.info('训练验证集模型结束')
    #         # xgb_model_v2.save_model('./user_data/model_data/xgb_model_v2.json')
    #         pickle.dump(xgb_model_v2, open('./user_data/model_data/xgb_model_v2.json', 'wb'))
    #         logger.info('模型文件保存在: ./user_data/model_data/xgb_model_v2.json')
    #     except:
    #         logger_error.error('训练验证集模型出错, 报错详细信息: ', traceback.format_exc())
    #         status = 0

