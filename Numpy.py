# 勉強元リンク: https://youtu.be/gnTxKHMYqFI?si=zjX2XeGGdkAHuCce

# NumPy
# 高度な数値計算ができるライブラリ
# Pythonのリストとは処理が異なる
# 配列同士の計算や、行列計算が可能

!pip install numpy
import numpy as np

# ndarray配列
# 1次元配列
# np.array(リスト)
# Pythonのリストとは処理が異なる

x_li = [10, 14, 19]
x_np = np.array([10, 14, 19])

print(x_li * 2)
print(x_np * 2)
y = np.array(["A", "B", "C"])

# 多次元の配列の作成

# 2次元配列
x_2d = np.array(
    [[10, 14, 15],
     [20, 24, 26]]
)

# 3次元配列
x_3d = np.array(
    [[[10, 11], [13, 14], [16, 17]],
     [[20, 21], [23, 24], [26, 27]]]
)

# 配列の操作
# 次元を調べる ndim
print(x_np.ndim, x_2d.ndim, x_3d.ndim)


# 各次元のサイズを調べる shape
print(x_np.shape, x_2d.shape, x_3d.shape)

# 初期化
# np.zeros(作りたい配列のshape)
# ones((サイズ1, サイズ2, ...))だとすべての要素を1で初期化する
# empty(サイズ1, サイズ2, ...)だと要素が空の配列　高速
# ramdom.rand(サイズ1, サイズ2, ...)だとすべての要素を0~1のランダムな値で初期化する

print(np.zeros(3))
print(np.ones((2,3)))
print(np.empty((2,4)))
print(np.random.rand(3,2))

# 配列の全要素に対する四則演算
# 足し算: 配列 + 数値
# 引き算: 配列 - 数値
# かけ算: 配列 * 数値
# わり算: 配列 / 数値

x = np.array(
    [[10, 14, 15],
     [20, 24, 26]]
)
print(x + 2)
print(x - 2)
print(x * 2)
print(x / 2)

# 同じshapeの配列同士の四則演算
# 演算は要素同士で行われる
# 足し算: 配列1 + 配列2
# 引き算: 配列1 - 配列2
# かけ算: 配列1 * 配列2
# わり算: 配列1 / 配列2

x = np.array(
    [[10, 14, 15],
     [20, 24, 26]]
)

y = np.array(
    [range(1,4),
     range(2,5)]
)

print(x + y)
print(x - y)
print(x * y)
print(x / y)

# 異なる大きさの配列で四則演算できる場合
# 1次元配列のサイズと
# 多次元配列の最後の次元のサイズが同じ

x = np.array(
    [[10, 14, 15],
     [20, 24, 26]]
)
y = np.array([2, 3, 4])

print(x + y)
print(x - y)
print(x * y)
print(x / y)

# n行1列の2次元配列の行サイズと
# 多次元配列の最後から2番目の次元のサイズが同じ

x = np.array(
    [[2],
     [3],
     [4]]
)

y = np.array(
    [[10, 11],
     [12, 13],
     [14, 15]]
)

print(x + y)
print(x - y)
print(x * y)
print(x / y)

# 行列の積計算
# 左側の配列の列サイズ (行数) と
# 右側の配列の行サイズ (列数) が同じ

x = np.array(
    [range(1,4),
     range(0,3)]
)

y = np.array(
    [[10, 11],
     [12, 13],
     [14, 15]]
)

print(np.dot(x,y))

# 配列の変形
# reshape(変形後のshape)
# 変形前と変形後の要素数が等しくないとエラーが出る

x = np.array(
    [range(1,4),
     range(0,3)]
)

print(x.reshape(3,2))

# 1次元配列に変形
# flatten()

x = np.array(
    [range(1,4),
     range(0,3)]
)

print(x.flatten())

# 配列の要素の取得
# 通常のリストと同じ

x = np.array(
    [range(1,4),
     range(0,3)]
)

print(x[0, :])
print(x[:, 1])
print(x[1, 1])
print(x[0:2, 0:2])


# 配列の結合
# np.concatenate([配列1, 配列2], 軸)
# 軸未指定だと縦に結合
# 1を指定すると横に結合

x = np.array(
    [range(1,4),
     range(0,3)]
)

y = np.array(
    [[10, 11],
     [12, 13],
     [14, 15]]
).reshape(2,3)

print(np.concatenate([x, y]))
print(np.concatenate([x, y], 1))

# 組み込み関数
# 最大値  : np.max(配列やリスト)
# 最小値  : np.min(配列やリスト)
# 合計    : np.sum(配列やリスト)
# 要素の積: np.prod(配列やリスト)
# 平均    : np.mean(配列やリスト)
# 標準偏差: np.std(配列やリスト)
# 分散    : np.var(配列やリスト)
# 中央値  : np.median(配列やリスト)

# 対数  : np.log(数値)
# 平方根: np.sqrt(数値)
# sin   : np.sin(数値)
# cos   : np.cos(数値)
# tan   : np.tan(数値)
# 円周率: np.pi