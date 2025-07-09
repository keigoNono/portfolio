# 勉強元リンクhttps://www.youtube.com/watch?v=tCMl1AWfhQQ&ab_channel=Python%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%9F%E3%83%B3%E3%82%B0VTuber%E3%82%B5%E3%83%97%E3%83%BC

print("こんにちは")

print("あいうえお")
print("かきくけこ")
print("さしすせそ")

# 1行の終端は改行か;
# ;はほぼ使わない
print("あいうえお")print("かきくけこ")
print("あいうえお");print("かきくけこ")

# 大文字と小文字は区別される
# 以下はエラーになる
Print("あいうえお")

# 長い計算式の改行は()で囲んで行う
(1000 + 5000 + 8000 + 9000 + 1300
 + 6000 + 25000 + 90000)

# コメントは#
# #以降は全てコメント
print("あいうえお") # コメントはこのようにも書ける

x = 10
x = 100
x = y # yには100が入る
# 変数名はどんなデータが入っているかわかるような名前に

apple_price = 200
print(apple_price) # 200と表示される

name = '斎藤'; # ""でもよい
weight = 54.5
# 変数の型は値が代入されたタイミングで自動で決まる

# 型によってできることが違う
# 同じ命令でも型によって動作が違う
x = 100
y = 200
z = x + y #zは300

x = '100'
y = '200'
z = x + y # zは'100200'

z = 100 + '100' # エラー

# 変数の型はtypeで調べる
print(type(x))

# 変数の型変換
x = '10'
y = int(x) # int型の10

apple_price = 100
a_type = type(apple_price)

name = '斎藤'
n_type = type(name)

weight = 54.5
w_type = type(weight)

print(apple_price, name, weight)
print(a_type, n_type, w_type)


math = 82
japanese = 74
english = 60

avg_score = (math + japanese + english) / 3
print(avg_score)

x = 100
y = 0
z = x / y # 0で割ったのでエラー
print(z)

surname = '佐々木'
given_name = 'まゆ'
full_name = surname + given_name
print(full_name)

# 変数埋め込み
price = 100
text = f"この商品は{price}円です"
print(text)

# リスト
student_names = ["斎藤", "小林", "佐々木", "田中"]

print(student_names[0]) # インデックスは0から始まる
print(student_names[2])
print(student_names[-1]) #インデックスが負の場合は後ろから数える
print(student_names[-3])


# リストの長さの取得
print(len(student_names))


# リストの分割
# リスト[開始位置:終了位置]
student_names = ["斎藤", "小林", "佐々木", "田中", "渡辺", "高橋"]
# 分割用の番号  0      1       2         3       4       5       6
#              -6     -5      -4        -3      -2      -1
print(student_names[1:3])
print(student_names[3:1]) # 開始位置 > 終了位置だとからのリストを返す
print(student_names[-3:-1])
print(student_names[-3:0]) # インデックスの指定に負の値と0は同時に使えない　からのリストを返す
print(student_names[:2]) # 開始位置の省略すると先頭から
print(student_names[3:]) # 終了位置を省略すると終端まで
print(student_names[:-4])
print(student_names[-4:])

# 空のリスト生成
x = []
y = list()

# リストの結合
x = ["a", "b"]
y = ["A", "B"]
print(x + y)

# 要素の追加、削除
x = [20, 12, 40]

#scores.append(92, 70) # エラー
x.append(92)
x.remove(12) # 削除したい 値 を引数に入れる

print(x)

# リストに格納できるもの
x = ["a", "b", "a"] # 同じものでも別々のものと解釈される
y = [1, 2, "a", "b"] # 型が違うものも入れられる


# 組み込み関数
x = [20, 12, 40]
print(sum(x))
print(sorted(x))
print(sorted(x, reverse = True))

# 辞書
# keyに対応するvalueを1つのペアとしてそのペアを複数持つもの

scores = {
    "数学" : 82,
    "国語" : 74,
    "英語" : 60,
    "理科" : 92,
    "社会" : 70
}

# 辞書の値の取得
science = scores["理科"]
science = scores.get("理科")

# 新たな要素の追加
prices = {
    "りんご" : 150,
    "バナナ" : 350,
    "りんご" : 170 # keyに同じものは存在できないため、後に指定したこちらだけ残る
}
prices["いちご"] = 560
print(prices)

# 要素の変更
prices["バナナ"] = 300
print(prices)

# 要素数の取得
print(len(prices))

# key, valueだけ取り出す
print(prices.keys())
print(prices.values())

x = list(prices.keys())
print(x)

# 空の辞書
x = {}
x = dict()

# 練習問題
# 1-1
# 以下で指定した辞書scoresをもとに、理科は社会より何点高いかを
# 「〇点」という文字で出力するプログラムを作ってください

scores = {
    "数学" : 82,
    "国語" : 74,
    "英語" : 60,
    "理科" : 92,
    "社会" : 70
}

# 解答
diff = scores["理科"] - scores["社会"]
print(f"{diff}点")

# 1-2
# 以下で指定した辞書scoresをもとに、点数の平均点を
# 「〇点」と出力するプログラムを作ってください

scores = {
    "数学" : 82,
    "国語" : 74,
    "英語" : 60,
    "理科" : 92,
    "社会" : 70
}

# 解答
scores_value = list(scores.values())
avg_score = sum(scores_value) / len(scores_value)
print(f"{avg_score}点")

# 別解
avg_score = sum(scores.values()) / len(scores.values())
print(f"{avg_score}点")

# 集合とタプル
# 集合　複数の値を1つにまとめたもの　順序がなく、同じ値を持てない
#                                    →インデックスがない
# 2つのグループがあったとき、両方にいるもの、片方にしかいないものの抽出に便利
x = {1, 2, 4} # 辞書はkeyとvalueのペア


# 要素の追加
x.add(7)
print(x)

# 要素の削除
x = {1, 2, 4}

x.discard(1)
x.remove(4)

print(x)

# 集合の要素にない値の指定
# x.discard(10) # エラーは出ない
# x.remove(10) # エラー

# 空の集合
x = set()

# 集合の操作
x = {0, 1, 3, 6}
y = {0, 2, 5, 6}

# 和集合
print(x | y)

# 差集合
print(x - y)

# 積集合
print(x & y)

# タプル
# 複数の値を1つにまとめたもの　順序を持ち、同じ要素を持てるが、値の変更が出来ない
# 関数を扱う中でよく出る
# 複数の要素で1つのデータとしてあらわされるようなもの
x = (1, 2, 4)

# 要素の抽出 0から始まる
print(x[1])

# インデント
# コードのブロックを表す
# コードを掻き始める位置を後ろに下げる
# 1インデント = 半角スペース4つ

# if文
# 条件分岐が書けるもの

# if 条件式1:
#     条件式1がTrueの時の処理
# elif 条件式2:
#     条件式1がFalseかつ条件式2がTrueの時の処理
# elif 条件式3:
#     条件式1, 2がFalseかつ条件式3がTrueの時の処理
# else:
#     条件式1,2, 3が全てFalseの時の処理

# 「値が同じである」という条件式
login_cnt = 1

if login_cnt == 1:
  print("初回ログインです")

# 「値が同じではない」という条件式

login_cnt = 2
if login_cnt != 1:
  print("初回ログインではないです")

# 値の大きさの比較
x = 1
y = 2
z = 2

if x < y:
  print("x < y")

if y <= z:
  if y < z:
    print("y < z")
  else:
    print("y = z")

# 「含まれる」という条件式
x = [1, 2, 3]
if 1 in x:
  print("1が含まれる")

# 条件式の組み合わせ
# A かつ B → A and B
# A または B → A or B

x = [2, 3, 6]
index = 2

if x[index] % 2 == 0 and x[index] % 3 == 0:
  print("xは6の倍数です")
elif x[index] % 2 == 0 or x[index] % 3 == 0:
  print("xは6の倍数ではない2の倍数か3の倍数です")

# 否定の条件式

x = [1, 2, 4]

if 5 not in x:
  print("xに5は含まれない")

if not (5 in x):
  print("xに5は含まれない")

# for文
# for 変数 in 繰り返しオブジェクト:
#     繰り返したい処理

# scores = [90, 30, 49]
scores = {90, 30, 49}
# scores = (90, 30, 49)

for x in scores:
  print(x)

# 辞書の場合の書き方
# for 変数1, 変数2 in 辞書.items():
#     繰り返したい処理

fruits = {
    "apple"  : 130,
    "banana" : 350,
    "lemon"  : 100
}

for name, price in fruits.items():
  print(f"{name}は{price}円です")

# for文でのrangeの使用
# range(整数) 0 から 整数 - 1 までの集まりを返す
# range(整数1, 整数2) 整数1 から 整数2 - 1 までの集まりを返す
# print(type(range(2))) # 型はrange

for x in range(5):
  print(x)

for x in range(1,5):
  print(x)

# break
# for文を途中で抜けたいときに使う

numbers = [10, 21, 100, 18, 2]

for n in numbers:
  if n >= 100:
    break
  print(n)

# continue
# for文の途中で後続処理をスキップして次の繰り返し処理を行いたい場合に使用する

numbers = [10, 21, 32, 65]

for n in numbers:
  if n % 2 == 0:
    continue
  print(n)

# 練習問題
# 2-1
# yearという変数に西暦年の数値が代入されています
# この西暦年がうるう年か判別して
# うるう年なら「うるう年です」平年なら「平年です」と
# 表示するプログラムを作ってください
# うるう年の定義
# 4で割り切れない年はうるう年ではない
# 4で割り切れる年はうるう年
# 100で割り切れる年はうるう年ではない
# 400で割り切れる年はうるう年

years = [3, 4, 100, 400]

for year in years:
  if year % 400 == 0:
    print("うるう年です")
  elif year % 100 == 0:
    print("平年です")
  elif year % 4 == 0:
    print("うるう年です")
  else:
    print("平年です")


# 練習問題
# FizzBuzz問題
# 1から100までの数字を出力させるプログラムを作りましょう
# ただし、
# 3の倍数の場合は数字の代わりにFizz
# 5の倍数の場合は数字の代わりにBuzz
# 15の倍数の場合は数字の代わりにFizzBuzz
# と表示させてください

for i in range(1,101):
  if i % 15 == 0:
    print("FizzBuzz")
  elif i % 5 == 0:
    print("Buzz")
  elif i % 3 == 0:
    print("Fizz")
  else:
    print(i)

# 関数
# 一連の処理を1つにまとめることができるもの
# def 関数の名前

# 関数定義
def print_hello():
  print("こんにちは")

# 関数呼び出し
print_hello()

# 注意 関数定義は関数呼び出しより前に書く

# 引数と戻り値
# 引数  : 関数に渡す値が代入された変数
# 戻り値: 関数が返す値

def add_numbers(a, b):
  c = a + b
  return c

print(add_numbers(10, 100))

def add_sub_numbers(a, b):
  c = a + b
  d = a - b
  return c, d # 返り値の型はタプル

print(add_sub_numbers(100, 30))

# キーワード引数
# 関数名(引数 = 値)

print(add_sub_numbers(b = 2, a = 3))

# 戻り値は複数可能

def is_leap_year(year):
  if year % 400 == 0:
    return True
  elif year % 100 == 0:
    return False
  elif year % 4 == 0:
    return True
  else:
    return False

year = 2024
print(is_leap_year(year))

# クラス
# オブジェクトを共通する属性 (データ) で分類したもの
# クラスがもつデータに対する処理も持たせる
# インスタンス変数: クラスがもつデータで、オブジェクトごとに異なる値を持つ変数
# メソッド: クラスがもつ処理　関数で定義される
# イニシャライザ インスタンス変数の初期化等の特別な処理を行う関数
# オブジェクトの作成時の1回だけ呼び出される
# self: オブジェクト自身のこと
# 慣習的に、クラス内で定義する関数の第1引数に
# クラス定義のメソッド内でのみ使用可能


class User:
  # イニシャライザ
  def __init__(self, name, mail_address, point):
    self.name = name
    self.mail_address = mail_address
    self.point = point

  def add_point(self, point):
    self.point += point




# オブジェクトの作成
# クラス名(イニシャライザに引数に渡す値)

user_1 = User("佐藤葵", "sato@exmaple.com", 500)
user_2 = User("小林ゆい", "kobayashi@exmaple.com", 1000)


# オブジェクトの要素の取得
print(user_1.name)
print(user_2.point)

# add_point()の確認

user_1.add_point(100)
print(user_1.point)

# 要素の値の変更

user_2.point = 0
print(user_2.point)

# 練習問題3
#アパレルのネット通販のアプリで商品クラスを作るケース

# インスタンス変数
# Id: id
# 商品名: name
# 販売価格: price
# 仕入れ価格: purchase_price

# メソッド
# 原価率 (仕入れ価格 /販売価格)

# 問題1 クラス定義を掻いて商品クラスのオブジェクトを作って原価率を表示
# 問題2 Tシャツのオブジェクトの販売価格を6000に変更して原価率を表示

# データ
# Id: A0001
# 商品名: 半袖クールTシャツ
# 販売価格: 5000
# 仕入れ価格 2250

class item:
  def __init__(self, id, name, price, purchase_price):
    self.id = id
    self.name = name
    self.price = price
    self.purchase_price = purchase_price

  def calc_COGS(self):
    return self.purchase_price / self.price

T_shirts = item("A0001", "半袖クールTシャツ", 5000, 2250)
print(T_shirts.calc_COGS())

T_shirts.price = 6000
print(T_shirts.calc_COGS())

# モジュール
# 別のPyhonスクリプトから利用されることを想定したファイル
# 主な中身は関数やクラスの定義
# 煩雑になった1つのファイルの中身を、関連してるものを切り出してモジュールにする
# 標準モジュール、標準ライブラリ: あらかじめ用意されているもの


# モジュールの使い方
# 同じファイル内にある場合
# from モジュール名 import 関数名・クラス名
# 複数インポートしたい場合は , で区切る
# インポート関数名の後に as 名前 で名前で使える

# サブファイルにある場合
# from サブディレクトリ名.モジュール名 import 関数名・クラス名
# 階層が深くなれば . でつないで表現

# 標準モジュールの使い方
# from モジュール名 import 関数名・クラス名

from datetime import datetime

t = datetime.today()
print(t)

# 外部ライブラリ
# ライブラリ: モジュールを複数まとめてパッケージ化したもの
# 外部ライブラリ: Python環境にインストールして使う
# matplotlib, openpyxl, pypdf など
# pipをコマンドラインツールで用いてインストールできる
# PyPIにコマンドがある

# pip install matplotlib

import matplotlib.pyplot as plt
label = ["A", "B", "C", "D"]
num = [20, 17, 25, 9]
plt.bar(label, num)
plt.savefig('./bar.png')

