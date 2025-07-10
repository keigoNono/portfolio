# 参考書籍: 池田雄太郎, 田尻俊宗, 新保雄大 (2023) 実務で役立つPython機械学習入門 課題解決のためのデータ分析の基礎, 翔泳社

# 必要なライブラリをインポート
def import_library():
  import numpy as np
  import pandas as pd
  import seaborn as sns
  from sklearn.metrics import mean_squared_error
  from sklearn.model_selection import train_test_split



# 訓練データとテストデータの作成
def gen_train_and_test_dataset():
  iris_df = sns.load_dataset("iris")
  # 説明変数の設定
  X = iris_df.loc[:,"sepal_width":"petal_width"]

  # 目的変数の設定
  y = iris_df["sepal_length"]

  # 訓練データとテストデータに分割
  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
  return train_X, test_X, train_y, test_y

# 予測誤差の算出
def calc_pred_error(X, y, model):
  # モデルによる予測
  pred = model.predict(X)

  # 予測誤差の算出　評価にはRMSEを使用
  pred_error = np.sqrt(mean_squared_error(pred, y))

  return pred_error

# 予測誤差の算出　LightGBM用
def calc_pred_error_for_lgb(X, y, model):
  # モデルによる予測
  pred = model.predict(X, num_iteration=model_LGB.best_iteration)

  # 予測誤差の算出　評価にはRMSEを使用
  pred_error = np.sqrt(mean_squared_error(pred, y))

  return pred_error

# 回帰アルゴリズム

# 1: 線形回帰
# 参考リンク: https://qiita.com/0NE_shoT_/items/08376b08783cd554b02e

# ライブラリの読み込み
import_library()
from sklearn.linear_model import LinearRegression

# データセットの生成
train_X, test_X, train_y, test_y = gen_train_and_test_dataset()

# モデルの初期化、学習
model_L = LinearRegression()
model_L.fit(train_X, train_y)

# 訓練誤差の算出
training_error = calc_pred_error(train_X, train_y, model_L)
print(training_error) # 0.306...

# テスト誤差の算出
test_error = calc_pred_error(test_X, test_y, model_L)
print(test_error) # 0.338...

# 2: リッジ回帰

# ライブラリの読み込み
import_library()
from sklearn.linear_model import Ridge

# データセットの生成
train_X, test_X, train_y, test_y = gen_train_and_test_dataset()

# モデルの初期化、学習
model_R = Ridge()
model_R.fit(train_X, train_y)

# 訓練誤差の算出
training_error = calc_pred_error(train_X, train_y, model_R)
print(training_error) # 0.299...

# テスト誤差の算出
test_error = calc_pred_error(test_X, test_y, model_R)
print(test_error) # 0.360...

# 3: 決定木
# 参考リンク: https://qiita.com/y_itoh/items/52064b6240fa9979b34e

# ライブラリの読み込み
import_library()
from sklearn.tree import DecisionTreeRegressor

# データセットの生成
train_X, test_X, train_y, test_y = gen_train_and_test_dataset()

# モデルの初期化、学習
model_D = DecisionTreeRegressor()
model_D.fit(train_X, train_y)

# 訓練誤差の算出
training_error = calc_pred_error(train_X, train_y, model_D)
print(training_error) # 0.034...

# テスト誤差の算出
test_error = calc_pred_error(test_X, test_y, model_D)
print(test_error) # 0.488...

# 4: ランダムフォレスト
# 参考リンク: https://qiita.com/mychaelstyle/items/48238d70c7602ca14a0c

# ライブラリの読み込み
import_library()
from sklearn.ensemble import RandomForestRegressor

# データセットの生成
train_X, test_X, train_y, test_y = gen_train_and_test_dataset()

# モデルの初期化、学習
model_RF = RandomForestRegressor()
model_RF.fit(train_X, train_y)

# 訓練誤差の算出
training_error = calc_pred_error(train_X, train_y, model_RF)
print(training_error) # 0.134...

# テスト誤差の算出
test_error = calc_pred_error(test_X, test_y, model_RF)
print(test_error) # 0.299...

# 5: 勾配ブースティング木 (LightGBM)
# 参考リンク: https://nuco.co.jp/blog/article/HZMvO2L4

# ライブラリの読み込み
import_library()
import lightgbm as lgb

# データセットの生成
train_X, test_X, train_y, test_y = gen_train_and_test_dataset()

# パラメータの設定
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',  # 回帰問題に
    'metric': 'rmse',  # Root Mean Squared Error (RMSE) を評価指標に設定
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1  # 情報出力レベルを最低に設定
}

# early_stoppingを使用するためのcallbackの設定
early_stopping = lgb.early_stopping(stopping_rounds=10)

# モデルの初期化、学習
model_LGB = lgb.train(params, lgb.Dataset(train_X, label=train_y), num_boost_round=100,
                  valid_sets=lgb.Dataset(test_X, label=test_y), valid_names=['sepal_length'],
                  callbacks=[early_stopping])

# 訓練誤差の算出
training_error = calc_pred_error_for_lgb(train_X, train_y, model_LGB)
print(training_error) # 0.350...

# テスト誤差の算出
test_error = calc_pred_error_for_lgb(test_X, test_y, model_LGB)
print(test_error) # 0.371...

# 5: ニューラルネットワーク
# 参考リンク: https://rinsaka.com/python/nn-regression.html

# ライブラリの読み込み
import_library()
!pip install keras
from keras.models import Sequential
from keras.layers import Dense

# データセットの生成
train_X, test_X, train_y, test_y = gen_train_and_test_dataset()

# モデルの初期化、学習
# モデル構造の定義
nn1 = 64
nn2 = 32
model_NN = Sequential()
model_NN.add(Dense(nn1, activation='relu', input_dim=3))
model_NN.add(Dense(nn2, activation='relu'))
model_NN.add(Dense(1, activation='linear'))

# モデルのコンパイル
model_NN.compile(optimizer='rmsprop',
          loss='mean_squared_error',
          metrics=['accuracy'])

# モデルの学習
model_NN.fit(train_X, train_y,
        batch_size=50,
        epochs=500,
        verbose=1)

# 訓練誤差の算出
training_error = calc_pred_error(train_X, train_y, model_NN)
print(training_error) # 0.269...

# テスト誤差の算出
test_error = calc_pred_error(test_X, test_y, model_NN)
print(test_error) # 0.277...
# 3000エポックで1時間かかった
# 500エポックなら1分 訓練誤差: 0.335, 汎化誤差: 0.302