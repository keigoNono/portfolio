# 勉強元リンク 前半 https://youtu.be/HYWQbAdsG6s?si=aveWU32solN_e9-Y
# 勉強元リンク 後半 https://youtu.be/lMt72Gggph4?si=YVayFx-mecQs4mgE

# pandas
# 前半
# データ分析に特化したライブラリ　表形式データ
# メリット:
# データ量上限 エクセルは100万行なのに対し、pandasはPCのメモリにのる範囲
# 処理スピード エクセルよりはるかに速い　データ量が多い時は特に推奨
# 再利用性     同じ操作を複数のファイルにしたいとき、pandasは一気に行える

# DataFrame
# 表形式データのクラス・オブジェクト
# カラム      : 列名
# インデックス: 行名
#              0 から始まる 一意でなくてよい

# Series
# 1 次元データのクラス・オブジェクト
# インデックスのみ持つ
# DataFrame を 1 行、1 列取り出したものは Series
import pandas as pd

# DataFrameの生成
# 書式
# pd.DataFrame({
# "列名 1" : ["要素 1", ...]
# ...
# })
# index = リスト で行名の指定
df = pd.DataFrame({
    "名前"  : [  "佐藤",   "斎藤",  "鈴木"],
    "年齢"  : [      21,     30 ,      18],
    "住所"  : ["東京都", "岐阜県", "埼玉県"],
    "血液型": [     "A",    "AB",     "O"]
},   index = [   "i-1",   "i-2",    "i-3"])

print(df)
# =>      名前  年齢   住所 血液型  
# => i-1  佐藤  21  東京都   A  
# => i-2  斎藤  30  岐阜県  AB 
# => i-3  鈴木  18  埼玉県   O 

# 新たな列の追加
df["身長"] = [160, 170, 180]
print(df)
# =>      名前  年齢   住所 血液型   身長
# => i-1  佐藤  21  東京都   A  160
# => i-2  斎藤  30  岐阜県  AB  170
# => i-3  鈴木  18  埼玉県   O  180

# Series の生成
# 書式
# pd.Series(リスト, index = リスト)
sr = pd.Series(  ["佐藤", "斎藤", "鈴木"],
    index = [ "i-1", "i-2",  "i-3"])

print(sr)
# => i-1    佐藤
# => i-2    斎藤
# => i-3    鈴木
# => dtype: object

# エクセルファイル読み込み
# read_excel("ファイルのパス")
# index_col  = "列名"    : インデックスとして扱いたい列の指定
# sheet_name = "シート名": 1 枚目以外のシートの参照

# エラー: Missing optional dependency 'xlrd'
# xlrd をインストールしても xlsx に対応しておらずエラー
# openpyxl を通じてもエラー: File is not a zip file
pd.read_excel("user_data.xlsx", index_col = "ユーザID", engine = "openpyxl") 

# csvファイル読み込み
# read_csv("ファイルのパス")
# index_col  = "列名": インデックスとして扱いたい列の指定

# 同様に
# json データ読み込み  : read.json
# xml データ読み込み   : read.xml
# pickle データ読み込み: read.pickle
# html データ読み込み  : read.html

# 特定の行や列を抽出
# 書式
# df.loc[インデックスのリスト, カラム名のリスト]
# df.iloc[行番号, 列番号]
# 全選択は : で表す
# 0 から始まる
df = pd.read_excel("user_data.xlsx", index_col = "ユーザID")
print(df.loc[["id001", "id003"], ["最高スコア"]])
print(df.loc["id002":"id004",:])
print(df.iloc[  [1], [1]])
print(df.iloc[[2,1], [1]]) # 指定した順番が表の順番と異なっていても問題ない

# 特定の列を抽出
# df[カラム名のリスト]
# 1 つだけの指定なら Series で返される
df = pd.read_excel("user_data.xlsx", index_col = "ユーザID")
print(df["平均スコア"])

# 抽出条件で操作
# query　今回は説明なし
# ブールインデックス
# True と False のリストで指定する
# 「かつ」は &, 「または」は |
# 複数の条件を指定するときは、条件を () で囲む
df = pd.read_excel("user_data.xlsx", index_col = "ユーザID")
print(df["最高スコア"] > 800)
print(df[df["最高スコア"] > 800])
print(df[(df["最高スコア"] > 800) & (df["平均スコア"] > 550)])

# 統計量を調べる
# NaN (非数) があるとそれを無視する
# 関数の引数に numeric_only = True ですべての数値の列を対象にできる
# 平均   : mean()
# 最大値 : max()
# 最小値 : min()
# 合計   : sum()
# 集計   : count()

# 標準偏差 : std() 不偏推定量
# 分散     : var()
# 中央値   : median()
# 歪度     : skew()
# 尖度     : kurt()
df = pd.read_excel("user_data.xlsx", index_col = "ユーザID")
print(df["平均スコア"].mean())
print(df[:].max())
print(df.min(numeric_only = True))

# 列単位で演算・加工
# df[列名1] 演算子 df[列名2]
# 返り値は Series
df = pd.read_excel("user_data.xlsx", index_col = "ユーザID")

print(df["最高スコア"] - df["平均スコア"])

df["名前"] = ["佐藤", "斎藤", "鈴木", "田中"]
df["敬称"] = df["名前"] + "さん"

print(df)

# 後編

# GroupBy
# 要素ごとにグループに分ける
# 分けたグループごとに処理を行うことを可能にする
df = pd.read_excel("deal_data.xlsx")
df.groupby("担当者").mean(numeric_only = True) # 引数なしだと日付も計算された

# インデックス名、カラム名の変更
# df.index   = [インデックス名のリスト]
# df.columns = [カラム名のリスト]
df = pd.read_excel("deal_data.xlsx")
df.index = range(1,10)
df.columns = ["date", "name", "sale"]

print(df)

# ある列をインデックスに設定
# df.set_index("列名")
# inplace = True を指定すると df 自身が変わる
df = pd.read_excel("deal_data.xlsx")
df_new = df.set_index("担当者")
print(df_new)

# インデックスを 0 から振りなおす
# df.reset_index()
# inplace = True を指定すると df 自身が変わる
# drop    = true を指定すると元のインデックスと置き換わる
df = pd.read_excel("deal_data.xlsx")
df_new = df.reset_index()
print(df_new)

# DataFrame の concat での結合
# 縦に結合
# pd.concat([df_1, df_2, ...])
# 要素がない部分は NaN が入る
df_1 = pd.DataFrame({
    "名前": [  "佐藤",   "斎藤",  "鈴木"],
    "年齢": [      21,     30 ,      18],
    "住所": ["東京都", "岐阜県", "埼玉県"],
})

df_2 = pd.DataFrame({
    "名前": [  "秋山",   "橋本"],
    "年齢": [      19,      51],
    "住所": ["大阪府", "千葉県"],
})

df_3 = pd.DataFrame({
    "名前": ["田中", "渡辺"],
    "年齢": [    40,    33]
})

print(pd.concat([df_1, df_2]))
# =>    名前  年齢   住所
# => 0  佐藤  21  東京都
# => 1  斎藤  30  岐阜県
# => 2  鈴木  18  埼玉県
# => 0  秋山  19  大阪府
# => 1  橋本  51  千葉県

print(pd.concat([df_1, df_3]))
# =>    名前  年齢   住所
# => 0  佐藤  21  東京都
# => 1  斎藤  30  岐阜県
# => 2  鈴木  18  埼玉県
# => 0  田中  40  NaN
# => 1  渡辺  33  NaN

# 横に結合
# concat([df_1, df_2, ...], axis = 1)
# 要素がない部分は NaN が入る
df_1 = pd.DataFrame({
    "名前": ["佐藤", "斎藤", "鈴木"],
    "年齢": [    21,   30 ,     18],
})
df_2 = pd.DataFrame({
    "住所"  : ["東京都", "岐阜県", "埼玉県"],
    "レベル": [     "A",     "B",      "S"]
})

print(pd.concat([df_1, df_2], axis = 1))
# =>    名前  年齢   住所 レベル
# => 0  佐藤  21  東京都   A
# => 1  斎藤  30  岐阜県   B
# => 2  鈴木  18  埼玉県   S

# DataFrame の merge での結合
# キーを基準に横に結合する
# merge(df_1, df_2, on = "key") # 3つ以上の同時結合は不可能
# how を指定しないと内部結合 (両方にあるデータのみを結合)
# how = "left"  で左外部結合 (左のキーは全て残して結合)
# how = "right" で右外部結合 (右のキーは全て残して結合)
# 同じ key がある場合の動作に注意
df_1 = pd.DataFrame({
    "id"  : ["000A", "000E", "000Q", "000Y"],
    "名前": [ "佐藤", "斎藤",  "鈴木", "藤井"],
    "年齢": [     21,   30 ,      18,    53],
})
df_2 = pd.DataFrame({
    "id"    : [ "000Q",   "000A",  "000E",   "000Z"],
    "住所"  : ["東京都", "岐阜県", "埼玉県", "広島県"],
    "レベル": [     "A",     "B",     "S",      "C"]
})

print(pd.merge(df_1, df_2, on = "id"))
# =>      id  名前  年齢   住所 レベル
# => 0  000A  佐藤  21  岐阜県   B
# => 1  000E  斎藤  30  埼玉県   S
# 2=>   000Q  鈴木  18  東京都   A

print(pd.merge(df_1, df_2, on = "id", how = "left"))
# =>      id  名前  年齢   住所  レベル
# => 0  000A  佐藤  21  岐阜県    B
# => 1  000E  斎藤  30  埼玉県    S
# => 2  000Q  鈴木  18  東京都    A
# => 3  000Y  藤井  53  NaN  NaN

print(pd.merge(df_1, df_2, on = "id", how = "right"))
# =>      id   名前    年齢   住所 レベル
# => 0  000Q   鈴木  18.0  東京都   A
# => 1  000A   佐藤  21.0  岐阜県   B
# => 2  000E   斎藤  30.0  埼玉県   S
# => 3  000Z  NaN   NaN  広島県   C

# map と lambda 式
# df["新たな列名"] = df["既存の列名"].map(lambda 式)
df = pd.DataFrame({
    "id"  : ["000A", "000E", "000Q", "000Y"],
    "年齢": [    21,    30 ,     18,     53],
})

df["区分"] = df["年齢"].map(lambda x: "成人" if x >= 20 else "未成年")

print(df)
# =>      id  年齢   区分
# => 0  000A  21   成人
# => 1  000E  30   成人
# => 2  000Q  18  未成年
# => 3  000Y  53   成人

# グラフの描画用ライブラリ
# !pip install japanize-matplotlib
import japanize_matplotlib

# グラフの描画
# 折れ線グラフの表示
# plot(x = "x 軸データの名前", y = "y 軸データの名前")
# kind = "bar" で棒グラフ
df = pd.DataFrame({
    "名前"   : ["佐藤", "斎藤", "鈴木", "藤井"],
    "年齢"   : [    21,   30 ,     18,     53],
    "購入額" : [  9000,  8200,   1200,   5000]
})

df.plot(x = "名前", y = "購入額")
df.plot(x = "名前", y = "購入額", kind = "bar")