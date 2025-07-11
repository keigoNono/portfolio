import pandas as pd
import japanize_matplotlib

df = pd.DataFrame({
    "名前"   : ["佐藤", "斎藤", "鈴木", "藤井"],
    "年齢"   : [    21,   30 ,     18,     53],
    "購入額" : [  9000,  8200,   1200,   5000]
})

df.plot(x = "名前", y = "購入額")
df.plot(x = "名前", y = "購入額", kind = "bar")
