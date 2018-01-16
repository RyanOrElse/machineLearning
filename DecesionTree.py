# coding:utf-8
from math import log
# 创建测试数据集


def create_data_set():
    data_set = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 分类属性
    return data_set, labels

# 数据集切分
def split_data_set(data_set, axis, value):
    ret_data_set = []
    for features in data_set:
        if features[axis] == value:
            reducedFeatVec = features[:axis]
            reducedFeatVec += features[axis+1:]
            ret_data_set.append(reducedFeatVec)
    return ret_data_set


def calc_ent(data_set):
    num = len(data_set)     # 返回数据集的行数
    label_count = {}        # 返回每个标签出现的次数
    for featureVec in data_set:
        current_label = featureVec[-1]
        if current_label not in label_count.keys():
            label_count[current_label] = 0      # 添加到字典
        label_count[current_label] += 1     # 计数
    ent = 0.0
    for key in label_count:
        prob = float(label_count[key]) / num
        ent -= prob * log(prob, 2)
    return ent

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calc_ent(dataSet)
    bestInfoGain = 0.0;
    bestFeatureIndex = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  #获取该列所有特征值
        uniqueVal = set(featList)
        newEnt = 0.0
        for value in uniqueVal:
            subDataSet = split_data_set(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEnt += prob * calc_ent(subDataSet)

        infoGain = baseEntropy - newEnt
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeatureIndex = i
    return bestFeatureIndex




if __name__ == '__main__':
    data_set, features = create_data_set()
    print(data_set)
    print(calc_ent(data_set))
    print(chooseBestFeatureToSplit(data_set))





