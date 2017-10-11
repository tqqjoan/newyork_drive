#-*- coding:UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import svm
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn import metrics
from math import sqrt
import holidays
import json
import urllib2
import time
class drive_duration(object):
    def __init__(self):
        # self.train_data = pd.read_csv(str)
        # print self.train_data.info()
        return
    def datetime_ft(self,ft):
        temp = pd.to_datetime(ft['pickup_datetime'])

        ft['month'] = temp.dt.month
        print temp
        # 提取星期特征
        ft['day'] = temp.dt.dayofweek

        # 提取时、分、秒特征
        ft['hour'] = temp.dt.hour
        ft['minute'] = temp.dt.minute
        ft['second'] = temp.dt.second

        # 是否是假期
        # for date, name in sorted(holidays.US(state='NY', years=2016).items()):
        #     print date, name
        l = temp.dt.date.as_matrix()
        h = holidays.US(state="NY")
        holi = []
        for i in l:
            if i in h:
                holi.append(1)
            else:
                holi.append(0)
        ft["holi"] = holi
        return ft
    def loc_ft(self,ft):
        #距离处理
        ft['x_dis'] = ft['dropoff_longitude'] - ft['dropoff_longitude']
        ft['y_dis'] = ft['dropoff_latitude'] - ft['dropoff_latitude']
        # square distance
        ft['dist_sq'] = (ft['y_dis'] ** 2) + (ft['x_dis'] ** 2)

        # distance
        ft['dist_sqrt'] = ft['dist_sq'] ** 0.5
        return ft
    def deal_ft(self,file):
        data = pd.read_csv(file)
        data = self.datetime_ft(data)
        data = self.loc_ft(data)
        #离散特征
        data_feature_cat = data[['vendor_id','day','holi']].as_matrix()
        print data_feature_cat

        # 连续特征
        data_feature_con = data[
            ['passenger_count','month', 'hour', 'minute', 'second', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
             'dropoff_latitude']].as_matrix()
        print data_feature_con  # 'x_dis','y_dis','dist_sqrt','dist_sq',

        # 对连续值特征标准化
        scaler = preprocessing.StandardScaler().fit(data_feature_con)
        data_feature_con_ed = scaler.transform(data_feature_con)

        # 对离散特征进行one_hot编码
        enc = preprocessing.OneHotEncoder()
        enc.fit(data_feature_cat)
        data_feature_cat_ed = enc.transform(data_feature_cat).toarray()

        # 将离散特征和连续特征组合

        data_feature = np.concatenate((data_feature_cat_ed, data_feature_con_ed), axis=1)
        return data_feature
    def get_label(self,file):
        data = pd.read_csv(file)
        # 提取训练数据Y
        y = data['trip_duration'].as_matrix()
        return y

    def rmsle(self,pred,y_test):
        sum = 0.0
        for x in range(len(pred)):
            p = np.log(abs(pred[x]) + 1)
            r = np.log(y_test[x] + 1)
            sum = sum + (p - r) ** 2
        return (sum / len(pred)) ** 0.5

    def project(self,str_train,str_test):
        #训练数据提取
        train_data = self.deal_ft(str_train)
        # 提取训练数据Y
        y = self.get_label(str_train)

        test_data = self.deal_ft(str_test)
        clf = svm.SVR(C=100000.0,gamma=100)
        t = int(sqrt(train_data.shape[1]))
        n = train_data.shape[0]
        # clf = RandomForestRegressor(n_estimators=100,max_features=t)
        # clf = linear_model.LinearRegression()
        # clf = GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,max_depth=10,random_state=0)
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for i in range(n):
            if i % 100 == 0:
                x_train.append(train_data[i,:])
                y_train.append(y[i])
            elif i % 100 == 1:
                x_test.append(train_data[i, :])
                y_test.append(y[i])
            else:
                continue
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        pred = pred.astype(int)
            # print y_train, pred
        scole = self.rmsle(pred, y_test)

        # clf.fit(data_feature[:1000,:],y[:1000])
        # pred = clf.predict(data_feature[:1000,:])
        # pred = pred.astype(int)
        # print y[:1000],pred
        # a = metrics.accuracy_score(y[:1000],pred)
        print scole
        # print(cross_validation.cross_val_score(clf,data_feature[:100000,:],y[:100000],cv=5))

        # return data_feature





if __name__ == '__main__':
    drive = drive_duration()
    drive.project("train_drive.csv","test_drive.csv")
