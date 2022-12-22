import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings
import gc
import spacy

pd.set_option('display.max_columns', 400)
pd.set_option('display.width', 400)
pd.set_option('display.max_rows', 400)


def grab_col_names(dataframe, cat_th=370, car_th=35):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
customer = pd.read_csv(r'C:\Users\izzet\PycharmProjects\DSMLBC9\Migros\customer.csv')
customeraccount = pd.read_csv(r'C:\Users\izzet\PycharmProjects\DSMLBC9\Migros\customeraccount.csv')
genel_kategoriler = pd.read_csv(r'C:\Users\izzet\PycharmProjects\DSMLBC9\Migros\genel_kategoriler.csv')
sample_submission = pd.read_csv(r'C:\Users\izzet\PycharmProjects\DSMLBC9\Migros\sample_submission.csv')
product_groups = pd.read_csv(r'C:\Users\izzet\PycharmProjects\DSMLBC9\Migros\product_groups.csv')
test = pd.read_csv(r'C:\Users\izzet\PycharmProjects\DSMLBC9\Migros\test.csv')
train = pd.read_csv(r'C:\Users\izzet\PycharmProjects\DSMLBC9\Migros\train.csv')
transaction_header = pd.read_csv(r'C:\Users\izzet\PycharmProjects\DSMLBC9\Migros\transaction_header.csv')
transaction_sale = pd.read_csv(r'C:\Users\izzet\PycharmProjects\DSMLBC9\Migros\transaction_sale.csv')

train['data'] = 'train'
test['data'] = 'test'
df = pd.concat([train, test])
df = pd.merge(df, genel_kategoriler, how='left', on='category_number')
#####################################
product_groups = pd.merge(product_groups, genel_kategoriler, how= 'left' , on= 'category_number')
product_groups["KategorikToplam"] = [str(row[0])+"_"+str(row[1])+"_"+str(row[2])+"_"+str(row[3])+"_"+str(row[4]) for row in product_groups.values]
product_groups["AltKategoriler"] = [str(row[1])+"_"+str(row[2])+"_"+str(row[3])+"_"+str(row[4]) for row in product_groups.values]
product_groups["Kategoriler"] = [str(row[5])+"_"+str(row[1])+"_"+str(row[2])+"_"+str(row[3])+"_"+str(row[4]) for row in product_groups.values]
product_groups = product_groups[['Kategoriler','AltKategoriler','category_number','genel_kategori']]
product_groups.loc[product_groups['AltKategoriler'].isin(['130_40_10_10', '130_70_30_20', '131_25_20_10', '210_10_25_10', '210_20_10_11',
 '210_20_10_13', '210_20_20_10', '210_20_20_11', '210_20_20_14', '210_20_20_15', '210_20_20_16',
 '210_20_20_17','220_50_50_10', '302_20_25_30','624_10_25_10','624_15_25_40','624_20_10_11','627_50_50_10']), 'Birden_Fazla_Kategorili_Eşya'] = 1
####################################
tr = pd.merge(transaction_sale, transaction_header, how= 'left' , on= 'basketid')
tr['kategori'] = [str(row[1])+"_"+str(row[2])+"_"+str(row[3])+"_"+str(row[4]) for row in tr.values]
tr = pd.merge(tr, customeraccount, how= 'left' , on= 'cardnumber')
tr = pd.merge(tr, customer, how= 'left' , on= 'individualnumber')
tr = pd.merge(product_groups, tr, right_on='kategori', left_on='AltKategoriler', how= 'left')
df = pd.merge(df,tr, how='left', on='individualnumber')
###################################
df['date_of_transaction'] = df['date_of_transaction'].apply(pd.to_datetime)
analysis_date = dt.datetime(2021,12,2)
###################################ü
###################################ü
#Yaş Değişkeni
###################################
df['Customer_Age'] = 2022 - df['dateofbirth']
##################################
#################################
#Tarihsel özellikler
df['day_of_month'] = df.date_of_transaction.dt.day
df["is_wknd"] = df.date_of_transaction.dt.weekday // 4
df['is_month_start'] = df.date_of_transaction.dt.is_month_start.astype(int)
df['is_month_end'] = df.date_of_transaction.dt.is_month_end.astype(int)
###################################
#Alışveriş Özellikleri
###################################
Sepettoplam = df.groupby('individualnumber')['basketid'].agg(['count'])
Kartsayısı = df.groupby('individualnumber').apply(lambda x: x.cardnumber.nunique())
Alışverişsayısı = df.groupby('individualnumber').apply(lambda x: x.basketid.nunique())
Kartsayısı = pd.DataFrame(Kartsayısı)
Alışverişsayısı = pd.DataFrame(Alışverişsayısı)
Müşteriyaşı = df.groupby('individualnumber').apply(lambda x: x.date_of_transaction.min())
Müşteriyaşı = pd.DataFrame(Müşteriyaşı)
df = pd.merge(df, Sepettoplam, how='left', on='individualnumber')
df['Sepettoplam'] = df['count']
df = pd.merge(df, Kartsayısı, how='left', on='individualnumber')
df['Kartsayısı'] = df[0]
df = pd.merge(df, Alışverişsayısı, how='left', on='individualnumber')
df['Alışverişsayısı'] = df['0_y']
df = pd.merge(df, Müşteriyaşı, how='left', on='individualnumber')
df['ilkalışveriş'] = df[0]
df['müşteri_en_eski_alışveriş'] = analysis_date - df['ilkalışveriş']
df['müşteri_en_eski_alışveriş'] = df['müşteri_en_eski_alışveriş'].dt.days
df.loc[(df["gender"]=='E'),"Erkek"] = 1
df.loc[(df["gender"]=='K'),"Kadın"] = 1

ab = df.groupby('individualnumber').agg({'amount' : ['mean','max','min','sum','median'],
                                                          'quantity' : ['mean','max','min','sum','median'],
                                                          'discount_type_1': ['mean','sum','size'],
                                                          'discount_type_2':['mean','sum','size'],
                                                          'discount_type_3':['mean','sum','size'],
                                                          'category_level_1': ['nunique'],
                                                          'category_level_2': ['nunique'],
                                                          'category_level_3': ['nunique'],
                                                          'category_level_4': ['nunique'],
                                         })

ab.columns = ['_'.join(col) for col in ab.columns]
ab.reset_index(inplace=True)

cd = df.groupby('individualnumber').agg({'is_sanal' : ['mean','size'],
                                                            'cardnumber': ['nunique'],
                                                            'basketid': ['nunique'],
                                                            'date_of_transaction': ['nunique'],
                                         })

cd.columns = ['_'.join(col) for col in cd.columns]
cd.reset_index(inplace=True)

df = pd.merge(df,ab, how='left', on='individualnumber')
df = pd.merge(df,cd, how='left', on='individualnumber')


a = ['category_number_x']
df = one_hot_encoder(df, a, drop_first=False)

df
df = df.fillna(0)
df.isnull().sum()
df = df[['individualnumber', 'category_number_x','hakkedis_amt','odul_amt','response','data','Birden_Fazla_Kategorili_Eşya','amount',
         'quantity','discount_type_1','discount_type_2','discount_type_3','is_sanal','kategori','city_code',
         'Customer_Age','day_of_month','is_wknd','is_month_start','is_month_end','Sepettoplam','Kartsayısı',
         'Alışverişsayısı','ilkalışveriş','müşteri_en_eski_alışveriş','Erkek','Kadın','amount_mean','amount_max',
         'amount_min','amount_sum','amount_median','quantity_mean','quantity_max','quantity_min','quantity_sum',
         'quantity_median','discount_type_1_mean','discount_type_1_sum','discount_type_1_size','discount_type_2_mean',
         'discount_type_2_sum','discount_type_2_size','discount_type_3_mean','discount_type_3_sum','discount_type_3_size',
         'category_level_1_nunique','category_level_2_nunique','category_level_3_nunique','category_level_4_nunique',
         'is_sanal_mean','is_sanal_size','cardnumber_nunique','basketid_nunique','date_of_transaction_nunique',
         'genel_kategori_x_diger','genel_kategori_x_gida','genel_kategori_x_hijyen','genel_kategori_x_icecek','genel_kategori_x_kisisel_bakim']]

y.value_counts()
df_train = df[df['data']=='train']
df_train = df_train.groupby('individualnumber').mean()
df_train = df_train.reset_index()
df_test = df[df['data']=='test']
df_test = df_test.groupby('individualnumber').mean()
df_test = df_test.reset_index()

label = "response"
except_cols = ["individualnumber",'response']

X = df_train.drop(["individualnumber",'response'], axis=1)
y = df_train['response']

df_train['response'].value_counts()




X_test = df_test.drop(["individualnumber",'response'], axis=1)

df_train.to_excel('df_train.xlsx')
df_train = pd.read_excel('df_train.xlsx')
#######################################
#####################################

xgboost_model = XGBClassifier(random_state=1308, scale_pos_weight=50)

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [3, 5, 18, 12, 15],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.2, 0.5]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

df_test['response'] = xgboost_final.predict(X_test)
df_test['response'].value_counts()

