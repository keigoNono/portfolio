# 参考書籍: 池田雄太郎, 田尻俊宗, 新保雄大 (2023) 実務で役立つPython機械学習入門 課題解決のためのデータ分析の基礎, 翔泳社

# データセットのロード
import seaborn as sns
planet_df = sns.load_dataset("planets")
iris_df = sns.load_dataset("iris")

# planet_df.head()
iris_df.head()
print(iris_df.describe())

# 必要なライブラリをインポート

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 回帰アルゴリズム

# 1: 線形回帰
# 参考リンク: https://qiita.com/0NE_shoT_/items/08376b08783cd554b02e

# モデルの読み込み
from sklearn.linear_model import LinearRegression


# 説明変数の設定
X = iris_df.loc[:,"sepal_width":"petal_width"]

# 目的変数の設定
y = iris_df["sepal_length"]

# 訓練データとテストデータに分割
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# モデルの初期化、学習
model_L = LinearRegression()
model_L.fit(train_X, train_y)

# 訓練データの予測
train_pred_sepal_length_L = model_L.predict(train_X)

# 訓練誤差の算出　評価にはRMSEを使用
print(np.sqrt(mean_squared_error(train_y, train_pred_sepal_length_L))) # 0.306...

# テストデータの予測
test_pred_sepal_length_L = model_L.predict(test_X)

# 訓練誤差の算出　評価にはRMSEを使用
print(np.sqrt(mean_squared_error(test_y, test_pred_sepal_length_L))) # 0.338...

# 2: リッジ回帰

# モデルの読み込み
from sklearn.linear_model import Ridge

# 説明変数の設定
X = iris_df.loc[:,"sepal_width":"petal_width"]

# 目的変数の設定
y = iris_df["sepal_length"]

# 訓練データとテストデータに分割
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# モデルの初期化、学習
model_R = Ridge()
model_R.fit(train_X, train_y)

# 訓練データの予測
train_pred_sepal_length_R = model_R.predict(train_X)

# 訓練誤差の算出　評価にはRMSEを使用
print(np.sqrt(mean_squared_error(train_y, train_pred_sepal_length_R))) # 0.299...

# テストデータの予測
test_pred_sepal_length_R = model_R.predict(test_X)

# 訓練誤差の算出　評価にはRMSEを使用
print(np.sqrt(mean_squared_error(test_y, test_pred_sepal_length_R))) # 0.360...

# 3: 決定木
# 参考リンク: https://qiita.com/y_itoh/items/52064b6240fa9979b34e

# モデルの読み込み
from sklearn.tree import DecisionTreeRegressor

# 説明変数の設定
X = iris_df.loc[:,"sepal_width":"petal_width"]

# 目的変数の設定
y = iris_df["sepal_length"]

# 訓練データとテストデータに分割
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# モデルの初期化、学習
model_D = DecisionTreeRegressor()
model_D.fit(train_X, train_y)

# 訓練データの予測
train_pred_sepal_length_D = model_D.predict(train_X)

# 訓練誤差の算出　評価にはRMSEを使用
print(np.sqrt(mean_squared_error(train_y, train_pred_sepal_length_D))) # 0.034...

# テストデータの予測
test_pred_sepal_length_D = model_D.predict(test_X)

# 訓練誤差の算出　評価にはRMSEを使用
print(np.sqrt(mean_squared_error(test_y, test_pred_sepal_length_D))) # 0.489...

# 4: ランダムフォレスト
# 参考リンク: https://qiita.com/mychaelstyle/items/48238d70c7602ca14a0c

# モデルの読み込み
from sklearn.ensemble import RandomForestRegressor

# 説明変数の設定
X = iris_df.loc[:,"sepal_width":"petal_width"]

# 目的変数の設定
y = iris_df["sepal_length"]

# 訓練データとテストデータに分割
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# モデルの初期化、学習
model_RF = RandomForestRegressor()
model_RF.fit(train_X, train_y)

# 訓練データの予測
train_pred_sepal_length_RF = model_RF.predict(train_X)

# 訓練誤差の算出　評価にはRMSEを使用
print(np.sqrt(mean_squared_error(train_y, train_pred_sepal_length_RF))) # 0.134...

# テストデータの予測
test_pred_sepal_length_RF = model_RF.predict(test_X)

# 訓練誤差の算出　評価にはRMSEを使用
print(np.sqrt(mean_squared_error(test_y, test_pred_sepal_length_RF))) # 0.299...

# 5: 勾配ブースティング木 (LightGBM)
# 参考リンク: https://nuco.co.jp/blog/article/HZMvO2L4

# モデルの読み込み
import lightgbm as lgb

# 説明変数の設定
X = iris_df.loc[:,"sepal_width":"petal_width"]

# 目的変数の設定
y = iris_df["sepal_length"]

# 訓練データとテストデータに分割
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

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

# 訓練データの予測
train_pred_sepal_length_LGB = model_LGB.predict(train_X, num_iteration=model_LGB.best_iteration)

# 訓練誤差の算出　評価にはRMSEを使用
print(np.sqrt(mean_squared_error(train_y, train_pred_sepal_length_LGB))) # 0.350...

# テストデータの予測
test_pred_sepal_length_LGB = model_LGB.predict(test_X, num_iteration=model_LGB.best_iteration)

# 訓練誤差の算出　評価にはRMSEを使用
print(np.sqrt(mean_squared_error(test_y, test_pred_sepal_length_LGB))) # 0.371...

# 5: ニューラルネットワーク
# 参考リンク: https://rinsaka.com/python/nn-regression.html

# モデルの読み込み
!pip install keras
from keras.models import Sequential
from keras.layers import Dense

# 説明変数の設定
X = iris_df.loc[:,"sepal_width":"petal_width"]

# 目的変数の設定
y = iris_df["sepal_length"]

# 訓練データとテストデータに分割
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

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
        epochs=3000,
        verbose=1)

# 訓練データの予測
train_pred_sepal_length_NN = model_NN.predict(train_X)

# 訓練誤差の算出　評価にはRMSEを使用
print(np.sqrt(mean_squared_error(train_y, train_pred_sepal_length_NN))) # 0.269...

# テストデータの予測
test_pred_sepal_length_NN = model_NN.predict(test_X)

# 訓練誤差の算出　評価にはRMSEを使用
print(np.sqrt(mean_squared_error(test_y, test_pred_sepal_length_NN))) # 0.277...
# 3000エポックで1時間かかった