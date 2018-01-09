import numpy as np
import csv

data = []
# 每个维度存储一种污染物的信息
for i in range(18):
    data.append([])

# read data
n_row = 0
with open('/Users/ryan/Downloads/train.csv', 'r', encoding='big5') as f:
    row = csv.reader(f)
    for i in row:
        # 第0行没有有效数据
        if n_row != 0:
            # 每一行只有第3-27格有值（每天24小时）
            for j in range(3, 27):
                if i[j] != 'NR':
                    data[(n_row - 1) % 18].append(float(i[j]))
                else:
                    data[(n_row - 1) % 18].append(float(0))
        n_row += 1
# parse data
x = []
y = []
# 每年12个月
for i in range(12):
    # 每个月每9个小时取一笔可以有471笔记录
    for j in range(471):
        x.append([])
        # 18种污染物
        for t in range(18):
            # 连续9小时
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s])
        y.append(data[9][480*i+j+9])

x = np.array(x)
y = np.array(y)


# add bias
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
# print(x.shape, y.shape)

# 初始化
w = np.zeros(x.shape[1])
print(w.shape)
l_rate = 1
repeat = 10000

# training
x_t = x.transpose()
s_gra = np.zeros(x.shape[1])

for i in range(repeat):
    hypo = np.dot(x, w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(loss)
    # cost_a = math.sqrt(cost)
    cost_a = np.sqrt(cost)
    gra = np.dot(x_t, loss)
    s_gra += gra ** 2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra / ada
    print('iteration:%d | cost:%f' % (i, cost_a))


# save mode
np.savetxt('model.txt', w)

# read mode
np.loadtxt('model.txt')

# read test data
test_x = []
n_row = 0
text = open('/Users/ryan/Downloads/test.csv', "r")
row = csv.reader(text, delimiter=",")

for r in row:
    if n_row % 18 == 0:
        test_x.append([])
        for i in range(2, 11):
            test_x[n_row//18].append(float(r[i]))
    else :
        for i in range(2, 11):
            if r[i] != "NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

# add square term
# test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1)


# get ans
ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w, test_x[i])
    ans[i].append(a)

filename = "/Users/ryan/Downloads/ans.csv"
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "value"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()











