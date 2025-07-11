import numpy   as np
import pandas  as pd
import seaborn as sns
import itertools

from sklearn.metrics         import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score

# RMSEを算出する
def RMSE(real, pred):
  return(np.sqrt(mean_squared_error(real, pred)))

# 予測誤差の算出
def calc_pred_error(X, y, model, metric):
  # モデルによる予測
  pred = model.predict(X)

  # 予測誤差の算出
  pred_error = metric(y, pred)

  return pred_error
# RMSEを算出する
def RMSE(real, pred):
  return(np.sqrt(mean_squared_error(real, pred)))

# 予測誤差の算出
def calc_pred_error(X, y, model, metric):
  # モデルによる予測
  pred = model.predict(X)

  # 予測誤差の算出
  pred_error = metric(y, pred)

  return pred_error

# 訓練データとテストデータの作成
def gen_dataset_for_classification():
  iris_df = sns.load_dataset("iris")
  # 説明変数の設定
  X = iris_df.loc[:, "sepal_length":"petal_width"]

  # 目的変数の設定
  y = iris_df["species"]

  # 訓練データとテストデータに分割
  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
  return train_X, test_X, train_y, test_y

# 文字列のラベルを数値に変換する
def label_to_int(label):
  iris_map = {
      "setosa"     : 0,
      "versicolor" : 1,
      "virginica"  : 2
  }
  return label.map(iris_map)

# accuracyの算出 多クラス分類のLightGBM用
def accuracy_score_for_LGB(X, y, model):
  # モデルによる予測
  pred = model.predict(X, num_iteration = model.best_iteration)

  # 予測誤差の算出
  pred_max   = [list(x).index(max(x)) for x in pred]
  pred_error = accuracy_score(y, pred_max)

  return pred_error

# 値を 0, 1, 2 の範囲で四捨五入する
# NNの分類の際のaccuracyの算出に使う
def discretization(num):
  if num > 1.5:
    return 2
  elif num > 0.5:
    return 1
  else:
    return 0

# accuracyの算出 多クラス分類のNN用
def accuracy_score_for_NN(X, y, model):
  # モデルによる予測
  pred = model.predict(X)

  # 予測誤差の算出
  pred_discretized = list(map(discretization, itertools.chain.from_iterable(pred)))
  pred_error       = accuracy_score(y, pred_discretized)

  return pred_error

# ライブラリの読み込み
# pip install keras
# pip install tensorflow
from keras.models import Sequential
from keras.layers import Dense

# データセットの生成
train_X, test_X, train_y, test_y = gen_dataset_for_classification()
train_y = label_to_int(train_y)
test_y  = label_to_int(test_y)

# モデルの初期化、学習
# モデル構造の定義
nn1 = 64
nn2 = 32
model_NN = Sequential()
model_NN.add(Dense(nn1, activation = 'relu', input_dim = 4))
model_NN.add(Dense(nn2, activation = 'relu'))
model_NN.add(Dense(  1, activation = 'linear'))

# モデルのコンパイル
model_NN.compile(
  optimizer = 'rmsprop',
  loss      = 'mean_squared_error',
  metrics   = ['accuracy'])

# モデルの学習
model_NN.fit(train_X, train_y,
  batch_size = 50,
  epochs     = 100,
  verbose    = 1)

# 訓練誤差の算出
training_error = accuracy_score_for_NN(train_X, train_y, model_NN)
print(training_error) # 0.959...

# テスト誤差の算出
test_error = accuracy_score_for_NN(test_X, test_y, model_NN)
print(test_error) # 1.0