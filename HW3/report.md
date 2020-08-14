# 分类实验报告

    这里是数据挖掘入门课程HW3的实验报告
    主要内容：分类问题引入、数据集与预处理、各种分类算法(使用库和手写)及其效率性能对比、Bagging方法的调参探究

----

## 分类问题 & 数据准备

分类任务即找一个函数判断输入数据所属的类别，输出为离散值。  
典型的分类问题：垃圾邮件识别、手写数字识别、人脸识别、语音识别。  

### 数据集概况

实验给出了10个数据集，字段类型混杂(`名称型nominal`/`数值型numeric`)，目标几乎全是是二分类，细节如下：

```python
dss = [
  # mixed, binary-classified
  'colic',          # 368 lns, 23 cols
  'credit-a',       # 690 lns, 16 cols
  'credit-g',       # 1000 lns, 21 cols
  'hepatitis',      # 155 lns, 20 cols
  # all numeric, binary-classified
  'breast-w',       # 699 lns, 10 cols
  'diabetes',       # 768 lns, 9 cols
  'mozilla4',       # 15545 lns, 6 cols
  'pc1',            # 1109 lns, 22 cols
  'pc5',            # 17186 lns, 39 cols
  # all numeric, trinary-classified
  'waveform-5000',  # 5000 lns, 41 cols
]
```

数据类型分为离散型和连续型分别对待：

- 对于**离散型**数据，用最大似然估计来建模它们的离散分布列
- 对于**连续型**数据，默认它们来自某个高斯分布；可用`均值二分`或`桶划分`将其离散化

注：本实验中，名称型nominal字段视作离散型数据，数值型numeric字段直接就是连续型数据

### 数据预处理

根据对数据集的初步观察，考虑以下几类预处理操作：

  - 转换编码：对于原始数据中离散的nominal型字段，有可能需要转换成整数
  - 处理缺失值：**删除**含NaN的列，或者用**列均值**填充
  - 正则化：对于数值型字段可以考虑**最大最小放缩MinMax**和**规范化Normalize**
  - 类型转换：对于连续字段，极差较大时可以考虑将其按**均值二分binarize**或者作**桶划分binning**从而变成离散字段

相关代码见`encode_literal()`、`deal_nan()`、`normalize()`、`binning()`函数。

### 训练数据划分

十折交叉验证：训练数据（随机打乱后）划分为十份，九份用于训练、一份用于测试，如此轮换十次，以模型度量的平均值来衡量模型的性能。代码如下：  

```python
def ten_fold(df, shuffle=True):
  if shuffle: df = df.sample(frac=1)  # 数据随机打乱
  silce = int(np.ceil(len(df) / 10))  # 每份分片大小
  ret = [ ]                           # 元素为二元组 (df_tr, df_ts)
  for i in range(10):
    cp1, cp2 = i*silce, (i+1)*silce   # 两个切分点
    df_ts = df[cp1:cp2]                      # 测试集占一份
    df_tr = pd.concat([df[:cp1], df[cp2:]])  # 剩下的是训练集
    ret.append((df_tr, df_ts))        # 组成一个fold
  return ret                          # 10个fold
```


## 分类算法(使用sklearn库)

首先使用sklearn库提供的现成算法跑通一个基线。

### 算法框架

由于sklearn库的封装非常优美，已经提供了大量常用的分类器模型，不同算法对外暴露同样的接口，因此**在每个数据集上依次使用每个训练器即可**，很容易写出统一的基本算法框架，入口代码如下：

```python
if __name__ == '__main__':
  # 各个数据集
  dss = ['colic', ...]           # 上文已列出，此略
  # 各个分类器
  clfs = [
    DecisionTreeClassifier(...), # Decision Tree
    BernoulliNB(...),            # Naive Bayes
    GaussianNB(...),
    SVC(...),                    # SVM
    LinearSVC(...),
    MLPClassifier(...),          # NN
    KNeighborsClassifier(...),   # kNN
    BaggingClassifier(...),      # Bagging (on kNN)
  ]
  
  # Lab1: using sklearn
  for ds in dss:                 # 对每个数据集训练每个分类器
    # 预处理
    df = get_data(ds)            # 加载数据集
    df = encode_literal(df)      # 将原始数据中的nominal类型字段编码为整型，这是大部分算法的要求
    df = deal_nan(df, 'fill')    # 处理缺失值NaN
    # 训练
    for clf in clfs:
      if 分类器是 SVM类 or MLP类:
        df = normalize(df)       # 对于SVM和神经网络，为了加速收敛而作规范化
      run(df, clf)               # 执行训练
```

训练和评估的核心部分`run(df, clf)`也很简单，即简单分为训练、测试、评估(precision/recall/fscore/AUC)三步，只需注意使用了十折交叉验证所以需要重复10次，代码如下：

```python
def run(df, clf):
  P, R, F, A = 0, 0, 0, 0             # 准确率、召回率、F值、AUC值

  for df_tr, df_ts in ten_fold(df):   # 十折交叉
    # 训练
    X, y = df_tr[df_tr.columns[:-1]], df_tr[df_tr.columns[-1]]
    clf = clf.fit(X, y)
    # 测试
    X, y = df_ts[df_ts.columns[:-1]], df_ts[df_ts.columns[-1]]
    y_pred = clf.predict(X)
    # 评估
    p, r, f, s = precision_recall_fscore_support(
                     y, y_pred, average='macro', zero_division=0)
    try: a = roc_auc_score(y, y_pred)
    except: a = 0                                # for non-binary
    P, R, F, A = P + p, R + r, F + f, A + a
  
  P, R, F, A = P * 10, R * 10, F * 10, A * 10    # 调整为百分比
  print('pres: %.2f%%, recall: %.2f%%, fscore: %.2f%%, AUC: %.2f%%' % (P, R, F, A))
  return P, R, F, A
```

### 性能效率对比

![运行时截图](screenshot/コメント%202020-05-31%20203217.png)

下面是上述8个分类器在10个数据集上的准确率P、召回率R、F值、AUC值表格：

| P/R/F/A(%) | colic | credit-a | credit-g | hepatitis | breast-w | diabetes | mozilla4 | pc1  | pc5  | waveform-5000 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| DecisionTree | 77.45/75.98/76.10/75.98 | 79.65/79.08/78.99/79.08 | 60.81/61.91/60.87/61.91 | 64.00/66.04/63.39/66.04 | 93.21/91.73/92.05/91.73 | 68.17/69.14/68.32/69.14 | 91.07/91.84/91.43/91.84 | 65.61/62.21/62.49/62.21 | 74.70/71.17/72.68/71.17 | 74.86/74.68/74.66/- |
| BernoulliNB  | 79.65/78.91/78.79/78.91 | 80.36/80.03/79.72/80.03 | 65.02/60.13/60.43/60.13 | 74.59/79.37/75.67/79.37 | 32.76/50.00/39.52/50.00 | 40.69/49.93/41.04/49.93 | 95.55/90.03/92.13/90.03 | 46.53/50.00/48.20/50.00 | 58.87/87.71/61.68/87.71 | 70.74/70.98/70.58/- |
|  GaussianNB  | 75.35/75.96/75.03/75.96 | 80.63/77.27/77.35/77.27 | 65.99/64.80/65.13/64.80 | 69.53/75.20/69.60/75.20 | 94.94/96.53/95.59/96.53 | 73.57/72.14/72.53/72.14 | 71.16/73.38/67.61/73.38 | 60.95/60.36/60.32/60.36 | 76.20/64.08/68.07/64.08 | 83.22/80.10/78.69/- |
|     SVC      | 87.50/84.06/84.50/84.06 | 85.50/86.27/85.32/86.27 | 71.11/60.48/60.70/60.48 | 78.78/72.45/71.59/67.76 | 95.91/96.43/96.14/96.43 | 74.75/71.29/72.00/71.29 | 85.44/86.48/85.91/86.48 | 66.78/54.81/55.75/54.81 | 81.35/59.38/63.73/59.38 | 86.15/86.11/86.03/- |
|  LinearSVC   | 79.68/79.86/79.54/79.86 | 86.94/86.56/86.06/86.56 | 66.18/59.64/59.40/59.64 | 77.99/75.07/72.94/75.07 | 96.26/96.29/96.21/96.29 | 76.36/72.65/73.48/72.65 | 82.11/82.08/82.09/82.08 | 73.84/56.20/58.06/56.20 | 81.54/60.11/64.76/60.11 | 86.81/86.86/86.77/- |
|   MLP(NN)    | 77.09/78.08/77.03/78.08 | 84.27/84.39/84.19/84.39 | 67.41/66.27/66.52/66.27 | 81.86/75.89/74.30/75.89 | 95.52/96.36/95.87/96.36 | 69.04/68.58/68.62/68.58 | 91.53/89.93/90.65/89.93 | 75.34/65.39/67.74/65.39 | 80.13/75.04/77.12/75.04 | 82.70/82.62/82.62/- |
|     kNN      | 77.64/78.47/77.42/78.47 | 82.98/82.50/82.45/82.50 | 66.04/59.95/60.26/59.95 | 75.28/70.97/70.35/70.97 | 96.51/96.03/96.16/96.03 | 70.98/68.85/69.10/68.85 | 87.93/87.94/87.93/87.94 | 71.58/64.18/65.70/64.18 | 79.57/73.56/76.10/73.56 | 77.43/77.47/77.40/- |
|   Bagging    | 81.64/79.31/79.51/79.31 | 85.26/83.77/83.89/83.77 | 71.10/55.27/52.43/55.27 | 67.38/62.20/62.06/62.20 | 96.23/96.63/96.38/96.63 | 70.76/66.72/67.28/66.72 | 92.61/89.73/90.92/89.73 | 72.85/54.95/56.64/54.95 | 81.78/67.52/72.28/67.52 | 82.49/82.34/82.09/- |

对应的时间开销：

| Time(s) | colic | credit-a | credit-g | hepatitis | breast-w | diabetes | mozilla4 | pc1 | pc5 | waveform-5000 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| DecisionTree | 0.0960 | 0.1080 | 0.1362 | 0.0870 | 0.0760 | 0.0770 | 0.5461 | 0.1170 | 0.8641 | 1.5790 |
| BernoulliNB  | 0.0970 | 0.0930 | 0.1220 | 0.0950 | 0.0690 | 0.0670 | 0.1110 | 0.0730 | 0.3340 | 0.1410 |
|  GaussianNB  | 0.0925 | 0.0920 | 0.1210 | 0.1060 | 0.0680 | 0.0555 | 0.1190 | 0.0690 | 0.3740 | 0.1360 |
|     SVC      | 0.1110 | 0.1540 | 0.3300 | 0.0680 | 0.0980 | 0.1740 | 24.943 | 0.2090 | 11.622 | 5.0158 |
|  LinearSVC   | 0.1880 | 0.2420 | 0.7271 | 0.0730 | 0.0820 | 0.3530 | 8.0243 | 4.6569 | 190.23 | 21.302 |
|   MLP(NN)    | 9.4259 | 25.465 | 39.112 | 4.0725 | 15.456 | 83.524 | 200.58 | 64.482 | 287.62 | 100.19 |
|     kNN      | 1.1106 | 1.1131 | 1.2431 | 1.1796 | 1.1056 | 1.1031 | 1.3836 | 1.1511 | 5.3299 | 1.6036 |
|   Bagging    | 0.2050 | 0.2400 | 0.4240 | 0.3051 | 0.2230 | 0.2130 | 1.3296 | 0.3250 | 26.447 | 5.5300 |

对比表格不难看出：

  - SVM类算法在几乎所有数据集上都有最好的表现，特别地LinearSVC在`pc5`上表现极差可知该数据集几乎线性不可分
  - BernoulliNB在多个数据集上表现极差，是因为并非所有字段都有较强的**二值化倾向**，但我们仍可见如数据集`mozilla4`上二值化模型的朴素贝叶斯效果略好一点
  - 朴素的多层感知机MLP**有些时候**在某些数据集上不容易收敛，这种炼金术特性提示我们可能需要针对性地设计网络结构细节
  - 此处的Bagging是基于kNN的，因此在大多数数据集上相比朴素kNN都有少量的性能提升，但数据集`hepatitis`却表现反常，暂且原因不明

至于时间开销，大多数算法都非常快，k近邻kNN由于其训练惰性而稍慢一点，多层感知机MLP(亦即NN)则慢很多，大概这就是炼金术叭（笑）


## 分类算法(手工实现)

然后试着自己尽力手工实现部分算法。  
由于各种分类算法的思想大相径庭，故详见各个小节。

### 决策树C4.5

决策树的一般模型：以原数据集为初始集合作为树根，考察某种**合理的划分**将该集合分成若干子集从而导致树的分叉，如此反复迭代，直到所有叶端集合的标签是纯的或满足一定容许误差时，决策树停止分叉。  
所谓“合理的划分”，可采用基于离散集合香农信息熵的`信息增益`(如ID3算法)或者`信息增益率`(如C4.5)来衡量。另由于子集划分必须是有限个，故对于连续型变量必须先做**离散化处理**。  

但是决策树的数据结构和分支条件推理实在是太繁琐了所以这个就跳过吧（笑）

### 朴素贝叶斯NB

朴素贝叶斯的一般模型：基于贝叶斯公式`P(A|B)P(B) = P(AB) = P(B|A)P(A)`，利用最大似然估计的想法最大化后验概率`P(A|B)`。对于离散数据，以给定的数据集建模其先验概率`P(B)`的分布列；对于连续数据，将给定的数据集拟合一个高斯分布。并假定每个特征维度是相互独立的 ，从而生成向量的概率为生成各分量概率的简单相乘。  
拉普拉斯修正：避免出现零概率，相乘的概率数的分子分母都加1。   

代码如下：
```python
def NaiveBayes(df):
  TP, FN, FP, TN = 0, 0, 0, 0    # confusion matrix
  for df_tr, df_ts in ten_fold(df):
    ytrain = df_tr[df_tr.columns[-1]]
    Xtest, ytest = df_ts[df_ts.columns[:-1]], df_ts[df_ts.columns[-1]]

    # train
    # P(Ci): probality of class Ci
    NCLASS = ytrain.value_counts().count()
    P_EPS = 1 / (NCLASS + 1)      # l'place adjust (?)
    Pc = [ytrain.value_counts()[i] / len(ytrain) for i in range(NCLASS)]
    # P(Aj|Ci) conditional probality of class Ci generating value Vk on attribute Aj
    Paoc = {i: [ ] for i in range(NCLASS)}    # in each list, index j respondes to the j-th attribute
    for idx, trgt in enumerate(ytrain.value_counts().keys()):
      Xtrain = df_tr[df_tr[df_tr.columns[-1]] == trgt]  # each subset of same class
      NSUBCLASS = len(Xtrain)
      for col in Xtrain:
        p = {k: (v + 1) / (NSUBCLASS + 1) for k, v in Xtrain[col].value_counts().items()}
        Paoc[idx].append(p)
    
    # test
    y_pred = [ ]
    for _, X in Xtest.iterrows():
      probs = [ ]
      for i in range(NCLASS):
        p = Pc[i]
        for idx, v in enumerate(X.values):
          try: p *= Paoc[i][idx][v]
          except KeyError: p *= P_EPS     # l'place adjust (?)
        probs.append(p)
      y_pred.append(np.argmax(probs))

    # analysis
    for i, y in enumerate(y_pred):
      if y == ytest.iloc[i]:
        if y == 1: TP += 1       # 1 = positive, 0 = negative
        else: TN += 1
      else:
        if y == 1: FP += 1
        else: FN += 1
  
  P, R, F = P_R_F(TP, FN, FP, TN)
  P, R, F = P * 100, R * 100, F * 100
  print('pres: %.2f%%, recall: %.2f%%, fscore: %.2f%%' % (P, R, F))
  return P, R, F
```

看起来实现得相当笨拙，实际上在测试集上也相当笨拙，关于拉普拉斯修正的部分不知道我有没有理解对（抱头

### 神经网络NN

感知机的一般模型：一个生物神经元的抽象模型，公式表述为`y = f(g(w,X),θ)`，其中`X`为输入向量，`g`为聚合函数、权向量`w`为`g`的参数，`f`为激活函数、阈值标量`θ`为`f`的参数，`y`为输出标量。  
神经网络的一般模型：多层感知机，从最朴素前馈的FNN密集连接到深层的DNN到带反馈的RNN、各种炼金术方法都有（笑）  

由于从底层开始手写NN不太现实，故使用了keras框架，代码如下：
```python
def NN(df):
  # model
  NFEAT, NCLASS = len(df.columns) - 1, 2
  model = Sequential([
    Dense(100, activation='relu', input_dim=NFEAT),
    Dense(100, activation='relu'),
    Dropout(0.25),
    Dense(NCLASS, activation='softmax'),
  ])
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

  # train
  TP, FN, FP, TN = 0, 0, 0, 0    # confusion matrix
  for df_tr, df_ts in ten_fold(df):
    Xtrain, ytrain = df_tr[df_tr.columns[:-1]], df_tr[df_tr.columns[-1]]
    Ytrain = to_categorical(ytrain, NCLASS)       # vectorize
    Xtest, ytest = df_ts[df_ts.columns[:-1]], df_ts[df_ts.columns[-1]]
    model.fit(Xtrain, Ytrain, batch_size=128, epochs=100)
    Ypred = model.predict(Xtest)
    for i, Y in enumerate(Ypred):
      y = np.argmax(Y)           # take the most probable class's id
      if y == ytest.values[i]:
        if y == 1: TP += 1       # 1 = positive, 0 = negative
        else: TN += 1
      else:
        if y == 1: FP += 1
        else: FN += 1
  
  P, R, F = P_R_F(TP, FN, FP, TN)
  P, R, F = P * 100, R * 100, F * 100
  print('pres: %.2f%%, recall: %.2f%%, fscore: %.2f%%' % (P, R, F))
  return P, R, F
```

提几个要点：

  - 网络模型使用了简单的三层密集连接，加入了一个Dropout层防止过拟合，事实上去掉该层甚至可以使得精确率达到95%以上
  - 因为是分类问题，故而选择了`categorical_crossentropy`作为损失函数，输出值也要做向量化
  - P、R、F三个度量值用简单计数的办法实现，计算公式见`P_R_F()`函数

### k近邻kNN

k近邻的一般模型：k个已知分类的最近邻居的(加权)投票，非常朴素的思想但有时性能最好。

代码如下：
```python
def kNN(df, k=5):
  TP, FN, FP, TN = 0, 0, 0, 0    # confusion matrix
  for df_tr, df_ts in ten_fold(df):
    Xtrain, ytrain = df_tr[df_tr.columns[:-1]], df_tr[df_tr.columns[-1]]
    Xtest, ytest = df_ts[df_ts.columns[:-1]], df_ts[df_ts.columns[-1]]

    y_pred = [ ]
    for _, centr in Xtest.iterrows():
      dist2 = ((Xtrain - centr)**2).sum(axis=1)
      dist_trgt = [(dist2.iloc[idx], ytrain.iloc[idx]) for idx, dist in enumerate(dist2.values)]
      dist_trgt.sort()                               # sort by distance
      trgts = [trgt for _, trgt in dist_trgt[:k]]    # only concern about k-nearest
      target = Counter(trgts).most_common(1)[0][0]   # label of most common
      y_pred.append(target)
    
    # analysis
    for i, y in enumerate(y_pred):
      if y == ytest.iloc[i]:
        if y == 1: TP += 1       # 1 = positive, 0 = negative
        else: TN += 1
      else:
        if y == 1: FP += 1
        else: FN += 1
  
  P, R, F = P_R_F(TP, FN, FP, TN)
  P, R, F = P * 100, R * 100, F * 100
  print('pres: %.2f%%, recall: %.2f%%, fscore: %.2f%%' % (P, R, F))
  return P, R, F
```

但不知为何这个实现非常低效、非常非常低效，可能`O(n^2)`算法确实就这么慢叭，暂时也不想优化了。

### 并行集成Bagging(基于kNN)

Bagging的一般模型：从原始数据集中**Bootstrap自采样**多个小数据集，在这些数据集上用多个同构或异构的分类器加以训练，预测时收集各个分类器给出的结果、综合评估后给出一个聚合值。

由于我们写出来的kNN实在是太慢，Bagging之后肯定慢得不行，所以暂时不写了 :(

### 性能效率对比

根据上一节中使用sklearn作基线跑出来的数据，我们选择原始数据的字段全是数值型、基线测试中F/R/P值不太高、数据量行列均较小的数据集`diabetes`来做对比测试。  
由于数据集`diabetes`的原始类型为数值型numerical，我们可以方便地将其视作连续型数据从而规范化，或者将其分桶从而离散化。我们将其离散化后进行**朴素贝叶斯NB**的实验，而将其视作连续值规范化后进行**神经网络NN**、**k近邻kNN**的实验。入口代码如下：

```python
if __name__ == '__main__':
  # Lab2: using handcraft
  df = get_data('diabetes')    # all numeric, binary-classified
  df = encode_literal_target(df)  # 编码目标标签到数字
  if True:                        # 作为离散值处理
    df_bin = binning(df, 10)
    NaiveBayes(df_bin)
  if True:                        # 作为连续值处理
    df_norm = normalize(df)
    NN(df_norm)
    kNN(df_norm)
```

下面是上述5个分类器在数据集`diabetes`上的准确率、召回率、F值、时间开销表格：

|   | Precision(%) | Recall(%) | F-Score(%) | Time(s) |
|:-:|:-:|:-:|:-:|:-:|
| NaiveBayes | 63.93 | 66.79 | 65.33 |  0.3066s |
|  DenseNN   | 85.06 | 82.84 | 83.93 | 15.3389s |
|    kNN     | 64.22 | 55.60 | 59.60 | 12.1949s |

对比表格行列不难看出：

  - 我们的手工实现相比库sklearn而言确实很劣质，甚至算法可能有些错误（狂笑），也可能是由于不应该离散化分桶而应该作为高斯分布来计算，但那个连续型俺不会写啊……
  - 但是手工构建的神经网络比sklearn默认的MLP效果好一些，甚至可以出奇地好、从而导致过拟合问题


## 专题：以kNN为基的Bagging方法的调参改进

Bagging方法的可控因素有：作为基的模型配置、分类器数量、抽样行列数、投票和聚合方式。  
在这里仍然使用sklearn的框架，使用在kNN上性能表现不太好的数据集`diabetes`来做对比测试，入口代码如下：

```python
if __name__ == '__main__':
  # Lab3: improve Bagging on kNN
  df = get_data('diabetes')
  df = encode_literal_target(df)  # 编码目标标签到数字
  kNN = KNeighborsClassifier      # 短别名
  clfs = [
    # alter k of kNN
    BaggingClassifier(base_estimator=kNN(3), n_estimators=5, max_samples=0.5),
    BaggingClassifier(base_estimator=kNN(5), n_estimators=5, max_samples=0.5),
    BaggingClassifier(base_estimator=kNN(7), n_estimators=5, max_samples=0.5),
    # alter n of clfs
    BaggingClassifier(base_estimator=kNN(5), n_estimators=3, max_samples=0.5),
    BaggingClassifier(base_estimator=kNN(5), n_estimators=5, max_samples=0.5),
    BaggingClassifier(base_estimator=kNN(5), n_estimators=10, max_samples=0.5),
    # alter r of sample
    BaggingClassifier(base_estimator=kNN(5), n_estimators=5, max_samples=0.4),
    BaggingClassifier(base_estimator=kNN(5), n_estimators=5, max_samples=0.6),
    BaggingClassifier(base_estimator=kNN(5), n_estimators=5, max_samples=0.8),
  ]
```

下面是上述5个分类器在数据集`diabetes`上的准确率、召回率、F值、AUC值、时间开销表格：

|   | Precision(%) | Recall(%) | F-Score(%) | AUC(%) | Time(s) |
|:-:|:-:|:-:|:-:|:-:|:-:|
| k = 3   | 68.87 | 67.15 | 67.22 | 67.15 | 0.1590 |
| k = 5   | 68.95 | 67.58 | 67.86 | 67.58 | 0.1981 |
| k = 7   | 71.91 | 69.44 | 69.98 | 69.44 | 0.2070 |
| n = 3   | 69.02 | 67.38 | 67.57 | 67.38 | 0.1120 |
| n = 5   | 70.40 | 68.24 | 68.63 | 68.24 | 0.1390 |
| n = 10  | 70.16 | 68.53 | 68.87 | 68.53 | 0.2310 |
| r = 0.4 | 70.32 | 69.03 | 68.82 | 69.03 | 0.1330 |
| r = 0.6 | 70.16 | 68.74 | 68.55 | 68.74 | 0.1530 |
| r = 0.8 | 68.16 | 67.23 | 67.08 | 67.23 | 0.1720 |

对比表格行列不难看出、在这个区间中，**准确度与k成正比、与n的函数形成虹拱、与r成反比**。因此我们根据这个线索修改k、n、r的中心，并进一步进行方格扫描：

|   | Precision(%) | Recall(%) | F-Score(%) | AUC(%) | Time(s) |
|:-:|:-:|:-:|:-:|:-:|:-:|
| k = 7   | 71.31 | 66.71 | 67.36 | 66.71 | 0.1500 |
| **k = 10**  | **70.23** | **65.77** | **66.24** | **65.77** | 0.1350 |
| k = 13  | 73.04 | 66.82 | 67.16 | 66.82 | 0.1400 |
| n = 4   | 71.86 | 66.40 | 66.98 | 66.40 | 0.1160 |
| **n = 5**   | **67.47** | **63.58** | **63.69** | **63.58** | 0.1350 |
| n = 6   | 71.84 | 66.48 | 66.87 | 66.48 | 0.1580 |
| r = 0.2 | 70.21 | 62.94 | 62.53 | 62.94 | 0.1290 |
| **r = 0.3** | **70.00** | 65.71 | 66.11 | 65.71 | 0.1400 |
| r = 0.4 | 70.94 | 67.44 | 67.61 | 67.44 | 0.1430 |

发生了比较意义不明的事：新选的中心点几乎都出现了**凹陷**，多次测试后仍然保持这个特征，暂且只能推断这就是数据集的特色，那就这样吧。

## 总结

这次实验学习了各种分类方法，思路各异缤纷如万华镜，隐于其下的数学推导还是有些苦手，特别是之后需要再详细考察SVM的数学基础。  
在实验数据集上的分类效果参差不齐，数据规模和参数设置可能是主要原因，这些剩余之物在实际的生产应用实践中仍有待练习。  

----

by Armit
2020/05/31 