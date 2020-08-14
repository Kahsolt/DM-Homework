#!/usr/bin/env python3
# Author: Armit
# Create Time: 2020/05/22 

import pdb
from os import path
import time
from io import StringIO
import rarfile

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, plot_tree
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical

BASE_DIR = path.dirname(path.abspath(__file__))
DATA_FILE = path.join(BASE_DIR, 'dataset', 'data.rar')
POSITIVE_TARGETS = [b'TRUE', b'true', b'1', b'tested_positive', b'malignant', b'DIE', b'bad', b'+', b'yes']

def timer(fn):
  def wrapper(*args, **kwargs):
    start = time.time()
    ret = fn(*args, **kwargs)
    end = time.time()
    print('[Timer]: %s() costs %.4fs' % (fn.__name__, end - start))
    return ret
  return wrapper

def P_R_F(TP, FN, FP, TN):
  P = TP / (TP + FP)
  R = TP / (TP + FN)
  F = 2 * TP / (2 * TP + FP + FN)
  return P, R, F

@timer
def run(df, clf):
  P, R, F, A = 0, 0, 0, 0

  for df_tr, df_ts in ten_fold(df):
    # train
    X, y = df_tr[df_tr.columns[:-1]], df_tr[df_tr.columns[-1]]
    clf = clf.fit(X, y)
    
    # test
    X, y = df_ts[df_ts.columns[:-1]], df_ts[df_ts.columns[-1]]
    y_pred = clf.predict(X)
    
    # analysis
    p, r, f, _ = precision_recall_fscore_support(
                     y, y_pred, average='macro', zero_division=0)
    try: a = roc_auc_score(y, y_pred)
    except: a = 0                                # for non-binary
    P, R, F, A = P + p, R + r, F + f, A + a
  
  P, R, F, A = P * 10, R * 10, F * 10, A * 10
  print('pres: %.2f%%, recall: %.2f%%, fscore: %.2f%%, AUC: %.2f%%' % (P, R, F, A))
  return P, R, F, A

@timer
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

@timer
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
    model.fit(Xtrain, Ytrain, batch_size=128, epochs=100, verbose=False)
    Ypred = model.predict(Xtest)
    for i, Y in enumerate(Ypred):
      y = np.argmax(Y)           # take the most probable class's id
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

@timer
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

def get_data(ds):
  rar = rarfile.RarFile(DATA_FILE, charset='utf8')
  try:
    with rar.open(f'data/{ds}.arff') as fh:
      bdata = fh.read()
    data = arff.loadarff(StringIO(bdata.decode()))
    return pd.DataFrame(data[0])
  except Exception as e:
    print('[Error]' + str(e)[:50])

def encode_literal_target(df):
  df = df.copy()
  col = df.columns[-1]    # last column is target label
  df[col] = df[col].apply(lambda x: (x in POSITIVE_TARGETS) and 1 or 0)
  return df

def encode_literal(df):
  df = df.copy()
  df = encode_literal_target(df)
  for col in df.columns[:-1]:
    if df.dtypes[col] == np.object:
      VALUES = list(set(df[col].values))  # literal str to int
      df[col] = df[col].apply(lambda x: (VALUES.index(x)))
  return df

def deal_nan(df, method='fill'):
  if not df.isna().sum().sum(): return df
  if method == 'fill':
    df = df.fillna(df.mean())
  elif method == 'drop':
    df = df.dropna(axis=0)
    df = df.dropna(axis=1)
  else: raise ValueError
  return df

def normalize(df):
  df = df.copy()
  for col in df.columns[:-1]:  # ignore taget label
    avg, std = df[col].mean(), df[col].std()
    df[col] = df[col].apply(lambda x: (x - avg) / std)
  return df

def binning(df, bins=5):
  df = df.copy()
  for col in df.columns[:-1]:  # ignore taget label
    if df.dtypes[col] == np.float64:
      dmin = df[col].min()
      gap = (df[col].max() - dmin) / bins
      df[col] = df[col].apply((lambda x: (x - dmin) // gap), convert_dtype=False)
  return df

def ten_fold(df, shuffle=True):
  if shuffle: df = df.sample(frac=1)
  silce = int(np.ceil(len(df) / 10))
  ret = [ ]    # (df_tr, df_ts)
  for i in range(10):
    cp1, cp2 = i*silce, (i+1)*silce
    df_ts = df[cp1:cp2]
    df_tr = pd.concat([df[:cp1], df[cp2:]])
    ret.append((df_tr, df_ts))
  return ret

def Lab1():
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
  
  clfs = [
    # Decision Tree
    DecisionTreeClassifier(
        criterion='gini',  # entropy
        splitter='best',   # random
        min_samples_split=3,
        min_samples_leaf=2),
    # Naive Bayes
    BernoulliNB(),
    GaussianNB(),
    # SVM
    SVC(max_iter=-1),
    LinearSVC(
        tol=1e-3,
        max_iter=10000),
    # NN
    MLPClassifier(max_iter=3000),
    # kNN
    KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',  # uniform
        algorithm='auto',
        p=2, # manhattan = 1, euclidean = 2
        n_jobs=4),
    # Bagging
    BaggingClassifier(
        base_estimator=KNeighborsClassifier(),
        n_estimators=10,
        max_samples=0.5,
        max_features=0.5),
  ]
  
  # Lab1: using sklearn
  print('[Lab1]: using sklearn')
  for ds in dss:
    print('=' * 72)
    print(f'>> [Dataset] using {ds}.')
    df = get_data(ds)
    df = encode_literal(df)
    df = deal_nan(df, 'fill')   # 'drop'
    for clf in clfs:
      clf_name = type(clf).__name__
      print(f'<< [Clf] using {clf_name}.')
      try:
        if 'SVC' in clf_name or 'MLP' in clf_name:
          df = normalize(df)
        run(df, clf)
      except Exception as e:
        print('[Error]' + str(e)[:50])
  print('*'*72)

def Lab2():
  # Lab2: using handcraft
  print('[Lab2]: using handcraft')
  df = get_data('diabetes')    # all numeric, binary-classified
  df = encode_literal_target(df)  # target label str to int
  print(f'>> [Dataset] using diabetes.')
  if True:                     # taken as discrete values
    df_bin = binning(df, 10)
    print(f'<< [Clf] using NaiveBayes.')
    NaiveBayes(df_bin)
  if True:                     # taken as continuous values
    df_norm = normalize(df)
    print(f'<< [Clf] using NN.')
    NN(df_norm)
    print(f'<< [Clf] using kNN.')
    kNN(df_norm)
  print('*'*72)

def Lab3():
  # Lab3: improve Bagging on kNN
  print('[Lab3]: improve Bagging on kNN')
  df = get_data('diabetes')
  df = encode_literal(df)
  kNN = KNeighborsClassifier      # short alias
  print(f'>> [Dataset] using diabetes.')
  clfs = [
    [
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
    ],
    [
      # alter k of kNN
      BaggingClassifier(base_estimator=kNN(7), n_estimators=5, max_samples=0.3),
      BaggingClassifier(base_estimator=kNN(10), n_estimators=5, max_samples=0.3),
      BaggingClassifier(base_estimator=kNN(13), n_estimators=5, max_samples=0.3),
      # alter n of clfs
      BaggingClassifier(base_estimator=kNN(10), n_estimators=4, max_samples=0.3),
      BaggingClassifier(base_estimator=kNN(10), n_estimators=5, max_samples=0.3),
      BaggingClassifier(base_estimator=kNN(10), n_estimators=6, max_samples=0.3),
      # alter r of sample
      BaggingClassifier(base_estimator=kNN(10), n_estimators=5, max_samples=0.2),
      BaggingClassifier(base_estimator=kNN(10), n_estimators=5, max_samples=0.3),
      BaggingClassifier(base_estimator=kNN(10), n_estimators=5, max_samples=0.4),
    ],
  ]
  for clf in clfs[0]:
    print(f'<< [Clf] using BaggingClassifier.')
    try:
      run(df, clf)
    except Exception as e:
      print('[Error]' + str(e)[:50])
  print('*'*72)

if __name__ == '__main__':
  Lab1()
  Lab2()
  Lab3()
