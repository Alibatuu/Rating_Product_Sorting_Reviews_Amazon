import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

##########################################
# Görev 1
##########################################

df = pd.read_csv("datasets/amazon_review.csv")
df.head()
df.describe().T
df.shape
# Adım 1
df["overall"].mean() # 4.587589013224822

# Adım 2
df["reviewTime"] = pd.to_datetime(df["reviewTime"])
df.dtypes
current_time = df["reviewTime"].max() # 2014-12-07
df["days_differance"] = (current_time - df["reviewTime"]).dt.days
df.head()
df["days_differance"].quantile([0.25,0.5,0.75]).T # 280, 430, 600
df.loc[df["days_differance"] <= 280, "overall"].mean() # 4.6957928802588995
df.loc[(df["days_differance"] > 280) & (df["days_differance"] <= 430), "overall"].mean() # 4.636140637775961
df.loc[(df["days_differance"] > 430) & (df["days_differance"] <= 600), "overall"].mean() # 4.571661237785016
df.loc[df["days_differance"] > 600, "overall"].mean() # 4.4462540716612375

df.loc[df["days_differance"] <= 280, "overall"].mean() * 38/100 + \
    df.loc[(df["days_differance"] > 280) & (df["days_differance"] <= 430), "overall"].mean() * 30/100 + \
    df.loc[(df["days_differance"] > 430) & (df["days_differance"] <= 600), "overall"].mean() * 22/100 + \
    df.loc[(df["days_differance"] > 600), "overall"].mean() * 10/100

# Adım 3
df["day_diff"].quantile([0.25,0.5,0.75]).T # 281, 431, 601
df.loc[df["day_diff"] <= 281, "overall"].mean() # 4.6957928802588995
df.loc[(df["day_diff"] > 281) & (df["day_diff"] <= 431), "overall"].mean() # 4.636140637775961
df.loc[(df["day_diff"] > 431) & (df["day_diff"] <= 601), "overall"].mean() # 4.571661237785016
df.loc[df["day_diff"] > 601, "overall"].mean() # 4.4462540716612375

def time_based_weighted_average(dataframe, w1=38, w2=30, w3=22, w4=10):
    return dataframe.loc[dataframe["day_diff"] <= 281, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 281) & (dataframe["day_diff"] <= 431), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 431) & (dataframe["day_diff"] <= 601), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 601), "overall"].mean() * w4 / 100
time_based_weighted_average(df)

# Adım 4
df.loc[df["days_differance"] <= 280, "overall"].mean()
df.loc[(df["days_differance"] > 280) & (df["days_differance"] <= 430), "overall"].mean()
df.loc[(df["days_differance"] > 430) & (df["days_differance"] <= 600), "overall"].mean()
df.loc[df["days_differance"] > 600, "overall"].mean()
# Zaman dilimlerine göre hesaplanan ortalama değerler bir önceki adımda görülmektedir.
# Hesaplanan değerlere bakıldığında ilk zaman diliminde puan ortalaması en yüksek değere sahiptir.
# Zaman arttıkça görülmektedir ki puan ortalaması azalmaktadır.
# Bu da göstermektedir ki daha yakın zamanda yapılan puanlamalar daha yüksek değerde olduğundan
# ürün şu anki trendlerde olan bir üründür. Üründe bir iyileştirme yapılmış olması da mümkündür.
# Geçmiş zamanda bu ürün,  şimdiye göre daha az beğenilmektedir.


##########################################
# Görev 2
##########################################

# Adım 1

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# Adım 2

def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

def score_pos_neg_diff(dataframe):
    dataframe["score_pos_neg_diff"] = dataframe.apply(lambda x: score_up_down_diff(x["helpful_yes"],
                                                                                 x["helpful_no"]), axis=1)

def score_average_rating_apply(dataframe):
    dataframe["score_average_rating"] = dataframe.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                                     x["helpful_no"]), axis=1)


def wilson_lower_bound_apply(dataframe):
    dataframe["wilson_lower_bound"] = dataframe.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                                   x["helpful_no"]), axis=1)

score_pos_neg_diff(df)
score_average_rating_apply(df)
wilson_lower_bound_apply(df)
df.head()

df.sort_values("wilson_lower_bound", ascending=False).head(10) # İlk 10
df.sort_values("wilson_lower_bound", ascending=True).head(10) # Son 10

# Wilson lower bound'a göre sıralanmış veriye bakıldığında, oylanma sayısı ve
# faydalı yorum sayısı yüksek olan yorumlar üst sıralarda görülmüştür.
# Wilson lower bound, verileri sıralarken yalnızca olumlu/olumsuz oylamalara bakmaz.
# Bu oylamalar haricinde, toplam oylanma sayısını da hesaba katar. Böylece en doğru değer
# bulunmuş olur. Son 10 yoruma bakıldığında, hiç oy almadığı görülmüştür ve bu beklenen bir durumdur.














