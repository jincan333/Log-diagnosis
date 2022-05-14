# -*- coding:utf-8 -*-


import pandas as pd
import re
import traceback
from datetime import datetime, date
import warnings
import logging
from logging import handlers
import sys
import os


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
    day_list = []
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
    label_df['time'] = label_df['fault_time']
    label_df['msg'] = ''
    label_df['server_model'] = label_df['sn'].map(dict(zip(log_df['sn'], log_df['server_model'])))
    label_df = label_df[['sn', 'time', 'msg', 'server_model', 'label']]
    log_label_df = pd.concat([log_df, label_df], axis=0).sort_values(by='time')
    log_label_df['fault_time'] = ''
    log_label_df = log_label_df[['sn', 'fault_time', 'msg', 'time', 'server_model', 'label']]
    logger.info('拼接日志和标签数据结束')

    logger.info('开始匹配日志和标签')
    logger.info('使用报错时间截断进行划分')
    # 使用报错时间截断进行划分
    FaultTime_log_correspond_label_df, FaultTime_no_label_log_list = divideLogByFaultTime(log_label_df)
    logger.info('匹配日志和标签结束')

    logger.info('开始计算统计特征')
    # 使用报错时间截断进行划分
    msg_log_label_df = calculateStatisticFeature(FaultTime_log_correspond_label_df)
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
    logger.info('数据条数: {}, 特征个数: {}'.format(feature_df.shape[0], feature_df.shape[1]-3))
    return feature_df


if __name__ == '__main__':
    # 调整根目录为项目主目录
    os.chdir(os.path.dirname(sys.path[0]))
    # 忽略warning
    warnings.filterwarnings("ignore")
    logger = get_logger('./user_data/logs/{}_info.log'.format(date.today()), 'info')
    logger_error = get_logger('./user_data/logs/{}_error.log'.format(date.today()), 'error')


    # 读取sel日志数据
    sel_log_df = pd.read_csv('./data/preliminary_sel_log_dataset.csv').drop_duplicates()
    # 读取训练标签数据：有重复数据！
    train_label1 = pd.read_csv('./data/preliminary_train_label_dataset.csv')
    train_label2 = pd.read_csv('./data/preliminary_train_label_dataset_s.csv')
    train_label_df = pd.concat([train_label1,train_label2],axis=0).drop_duplicates()
    # 读取A榜测试集
    preliminary_sel_log_dataset_a = pd.read_csv('./data/preliminary_sel_log_dataset_a.csv')
    preliminary_submit_dataset_a = pd.read_csv('./data/preliminary_submit_dataset_a.csv')
    # 读取B榜测试集
    preliminary_sel_log_dataset_b = pd.read_csv('./data/preliminary_sel_log_dataset_b.csv')
    preliminary_submit_dataset_b = pd.read_csv('./data/preliminary_submit_dataset_b.csv')
    # 读取词列表
    v1_word_list = list(pd.read_csv('./user_data/words/word_frequency_df.txt',sep='\t')['word'])
    v1p1_word_list = list(pd.read_csv('./user_data/words/tags_incomplete.txt',sep='\t',names=['word'])['word'])
    word_list = list(set(v1_word_list+v1p1_word_list))
    feature_df = caculateFeature(sel_log_df, train_label_df, word_list)
    feature_df.to_csv('./user_data/feature_data/xgb_model_v2_test.csv', index = None)
    logger.info('特征结果保存在: ./user_data/feature_data/xgb_model_v2.csv')




    # status = 1
    # # 读取模型训练数据
    # if status == 1:
    #     try:
    #         # 读取sel日志数据
    #         sel_log_df = pd.read_csv('./data/preliminary_sel_log_dataset.csv').drop_duplicates()
    #         # 读取训练标签数据：有重复数据！
    #         train_label1 = pd.read_csv('./data/preliminary_train_label_dataset.csv')
    #         train_label2 = pd.read_csv('./data/preliminary_train_label_dataset_s.csv')
    #         train_label_df = pd.concat([train_label1,train_label2],axis=0).drop_duplicates()
    #         # 读取A榜测试集
    #         preliminary_sel_log_dataset_a = pd.read_csv('./data/preliminary_sel_log_dataset_a.csv')
    #         preliminary_submit_dataset_a = pd.read_csv('./data/preliminary_submit_dataset_a.csv')
    #         # 读取B榜测试集
    #         preliminary_sel_log_dataset_b = pd.read_csv('./data/preliminary_sel_log_dataset_b.csv')
    #         preliminary_submit_dataset_b = pd.read_csv('./data/preliminary_submit_dataset_b.csv')
    #     except:
    #         logger_error.error('读取模型训练数据报错, 报错详细信息: ', traceback.format_exc())
    #         status = 0
    #
    # # 匹配日志和标签
    # if status == 1:
    #     try:
    #         sel_log_df['label'] = ''
    #         train_label_df['time'] = train_label_df['fault_time']
    #         train_label_df['msg'] = ''
    #         train_label_df['server_model'] = train_label_df['sn'].map(dict(zip(sel_log_df['sn'],sel_log_df['server_model'])))
    #         train_label_df = train_label_df[['sn', 'time', 'msg', 'server_model', 'label']]
    #         log_label_df = pd.concat([sel_log_df,train_label_df], axis = 0).sort_values(by = 'time')
    #         log_label_df['fault_time'] = ''
    #         log_label_df = log_label_df[['sn', 'fault_time', 'msg', 'time', 'server_model', 'label']]
    #
    #         logger.info('开始匹配训练集日志和标签')
    #         logger.info('使用报错时间截断进行划分')
    #         # 使用报错时间截断进行划分
    #         FaultTime_log_correspond_label_df, FaultTime_no_label_log_list = divideLogByFaultTime(log_label_df)
    #         logger.info('匹配训练集日志和标签结束')
    #     except:
    #         logger_error.error('匹配训练集日志和标签出错, 报错详细信息: ', traceback.format_exc())
    #         status = 0
    #
    # # 计算统计特征
    # if status == 1:
    #     try:
    #         logger.info('开始计算统计特征')
    #         # 使用报错时间截断进行划分
    #         msg_log_label_df = calculateStatisticFeature(FaultTime_log_correspond_label_df)
    #         logger.info('计算统计特征结束')
    #     except:
    #         logger_error.error('计算统计特征出错, 报错详细信息: ', traceback.format_exc())
    #         status = 0
    #
    # # 计算词频特征
    # if status == 1:
    #     try:
    #         msg_log_list = list(msg_log_label_df['msg_log'])
    #         label_list = list(msg_log_label_df['label'])
    #         v1_word_list = list(pd.read_csv('./user_data/words/word_frequency_df.txt',sep='\t')['word'])
    #         v1p1_word_list = list(pd.read_csv('./user_data/words/tags_incomplete.txt',sep='\t',names=['word'])['word'])
    #         v1p2_word_list = list(set(v1_word_list+v1p1_word_list))
    #         # 计算词频向量
    #         logger.info('开始计算词频特征')
    #         frequency_vector_list = []
    #         tag = 0
    #         for word in v1p2_word_list:
    #             if tag % 100 == 0:
    #                 print(tag, datetime.now())
    #             pattern = re.compile(word)
    #             frequency_vector = [len(re.findall(pattern, log)) for log in msg_log_list]
    #             frequency_vector_list.append(frequency_vector)
    #             tag += 1
    #         logger.info('计算词频特征结束')
    #     except:
    #         logger_error.error('计算词频特征出错, 报错详细信息: ', traceback.format_exc())
    #         status = 0
    #
    # # 拼接统计特征和词频特征
    # if status == 1:
    #     try:
    #         frequency_vector_df = pd.DataFrame(frequency_vector_list)
    #         frequency_vector_df = frequency_vector_df.T
    #         frequency_vector_df.columns = v1p2_word_list
    #         new_feature_list = list(msg_log_label_df.columns)[2:-2]
    #         frequency_vector_df[new_feature_list] = msg_log_label_df[new_feature_list]
    #
    #         frequency_vector_df['label'] = label_list
    #         frequency_vector_df[['sn', 'fault_time']] = msg_log_label_df[['sn', 'fault_time']]
    #     except:
    #         logger_error.error('拼接统计特征和词频特征出错, 报错详细信息: ', traceback.format_exc())
    #         status = 0
    #
    # if len(frequency_vector_df) > 0:
    #     frequency_vector_df.to_csv('./user_data/feature_data/xgb_model_v2.csv', index = None)
    #     logger.info('特征结果保存在: ./user_data/feature_data/xgb_model_v2.csv')




