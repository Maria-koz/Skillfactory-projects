#!/usr/bin/env python
# coding: utf-8

# # Импорт библиотек и создание функций

# In[515]:


import numpy as np
import pandas as pd

import re
import sys
import itertools
import datetime
from tqdm.notebook import tqdm
import pandas_profiling

from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer

from catboost import CatBoostRegressor
import xgboost as xgb

from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope

import warnings
warnings.filterwarnings("ignore")


# In[516]:


print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)


# In[517]:


# зафиксируем версии пакетов, чтобы эксперименты были воспроизводимы:
get_ipython().system('pip freeze > requirements.txt')


# In[518]:


# зафиксируем RANDOM_SEED, чтобы эксперименты были воспроизводимы:
RANDOM_SEED = 42


# ### Функции предобработки

# In[519]:


# заполнение engineDisplacement числовыми значениями
def mape(y_true: np.ndarray,
         y_pred: np.ndarray):
    return np.mean(np.abs((y_pred-y_true)/y_true))

# заполнение owners числовыми значениями
def transf_engineDisplacement_to_float(row: str):
    extracted_value = re.findall('\d\.\d', str(row))
    if extracted_value:
        return float(extracted_value[0])
    return None

# заполнение vehicleTransmission числовыми значениями
def transf_owners_to_float(value: str):
    if isinstance(value, str):
        return float(value.replace('\xa0', ' ').split()[0])
    return value

# заполнение enginePower числовыми значениями
def transf_vehicleTransmission_to_categ(value: str):
    if isinstance(value, str):
        if value in ['MECHANICAL', 'механическая']:
            return 'mechanical'
        else:
            return 'automatic'
    return value



def transf_enginePower_to_float(value: str):
    if isinstance(value, str):
        if value == 'undefined N12':
            return None
        else:
            return float(value.replace(' N12', ''))
    return value


def vis_num_feature(data: pd.DataFrame,
                    column: str,
                    target_column: str,
                    query_for_slicing: str):
    # построение графиков для численных переменных
    plt.style.use('seaborn-paper')
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    data[column].plot.hist(ax=ax[0][0])
    ax[0][0].set_title(column)
    sns.boxplot(data=data, y=column, ax=ax[0][1], orient='v')
    sns.scatterplot(data=data.query(query_for_slicing),
                    x=column, y=target_column, ax=ax[0][2])
    np.log2(data[column] + 1).plot.hist(ax=ax[1][0])
    ax[1][0].set_title(f'log2 transformed {column}')
    sns.boxplot(y=np.log2(data[column]), ax=ax[1][1], orient='v')
    plt.show()


def calculate_stat_outliers(data_initial: pd.DataFrame,
                            column: str,
                            log: bool = False):

    data = data_initial.copy()
    if log:
        data[column] = np.log2(data[column] + 1)
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    IQR = q3 - q1
    mask25 = q1 - IQR * 1.5
    mask75 = q3 + IQR * 1.5

    values = {}
    values['borders'] = mask25, mask75
    values['# outliers'] = data[(
        data[column] < mask25)].shape[0], data[data[column] > mask75].shape[0]
    return pd.DataFrame.from_dict(data=values, orient='index', columns=['left', 'right'])


def show_boxplot(data: pd.DataFrame,
                 column: str,
                 target_column: str):
    # построение боксплотов для численных признаков
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.boxplot(x=column, y=target_column,
                data=data.loc[data.loc[:, column].isin(
                    data.loc[:, column].value_counts().index)],
                ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + column)
    plt.show()


# # DATA

# ### Парсинг

# In[520]:


# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# import time
# import requests as r
# import wget
# import gzip
# import xml.etree.ElementTree as ET

# # Читаем файл robots.txt с сайта https://auto.ru/robots.txt и получаем значение параметра Sitemap
# url = "https://auto.ru/robots.txt"
# response = r.get(url)
# url = response.text[response.text.find("Sitemap"):][9:response.text[3625:].find("\n")]
# response = r.get(url)

# # Парсим полученный xml
# root = ET.fromstring(response.text)

# # Добаваляем архивы предложений автомобилей со строками "offers_cars" в массив
# look_for_string = "offers_cars"
# offers_arc = []

# for element in root:
#     if look_for_string in element[1].text: offers_arc.append(element[1].text)

# # Скачиваем архивы с объявлениями
# for arc in offers_arc:
#     wget.download(arc)

# # Создадим список ссылок на объявления
# pages_url_list=[]

# # Открываем и читаем архивы
# for arc in offers_arc:
#     file_name = arc[arc.find("sitemap_offers_cars"):]

#     with gzip.open(file_name, 'rb') as f:
#         file_content = f.read()

#     # Файл внутри представляет собой XML. парсим его
#     root = ET.fromstring(file_content)

#     # Ищем строки, содержашие used - авто с пробегом и добавляем их в массив ссылок
#     look_for_string = "used"
#     for element in root:
#         if look_for_string in element[0].text: pages_url_list.append(element[0].text)

# # Определяем итоговый массив данных
# cars = []

# # Запускаем Selenium
# chrome_options = Options()
# driver = webdriver.Chrome(executable_path=r"chromedriver.exe",options=chrome_options)

# # Открываем каждую ссылку, заносим данные в массив
# for url in pages_url_list:
#     driver.get(url)
#     #Определяем бренд и модель
#     elements = driver.find_elements(By.CLASS_NAME,'CardBreadcrumbs__itemText')
#     brand = elements[3].text
#     model_name = elements[4].text

#     # Определяем кузов
#     element =  driver.find_elements(By.CLASS_NAME,'CardInfoRow_bodytype')
#     bodyType = element[0].text.replace("\n"," ")

#     # Определяем цвет
#     element =  driver.find_elements(By.CLASS_NAME,'CardInfoRow_color')
#     color = element[0].text.replace("\n"," ").replace("Цвет ","")

#     # Определяем параметры двигателя
#     element =  driver.find_elements(By.CLASS_NAME,'CardInfoRow_engine')
#     engine = element[0].text.replace("Двигатель","").split("/")

#     # Получаем значение лошадиных сил, мощность и тип топлива
#     engineDisplacement = engine[0][1:].replace("л ","LTR")
#     enginePower = engine[1][1:].replace("л.с. ", "N12")
#     fuelType = engine[2]

#     # Определяем пробег
#     element =  driver.find_elements(By.CLASS_NAME,'CardInfoRow_kmAge')
#     mileage = element[0].text.replace("Пробег\n","").replace(" км","").replace(" ","")

#     # Определяем год выпуска
#     element =  driver.find_elements(By.CLASS_NAME,'CardInfoRow_year')
#     productionDate = element[0].text.split("\n")[1]

#     # Определяем трансмиссию
#     element = driver.find_elements(By.CLASS_NAME,'CardInfoRow_transmission')
#     vehicleTransmission = element[0].text.split("\n")[1]

#     # Определяем владельца
#     element = driver.find_elements(By.CLASS_NAME,'CardInfoRow_ownersCount')
#     owners = element[0].text.split("\n")[1]

#     # Определяем ПТС
#     element = driver.find_elements(By.CLASS_NAME,'CardInfoRow_pts')
#     pts = element[0].text.split("\n")[1]

#     # Определяем привод
#     element = driver.find_elements(By.CLASS_NAME,'CardInfoRow_drive')
#     wd = element[0].text.split("\n")[1]

#     # Определяем руль
#     element = driver.find_elements(By.CLASS_NAME,'CardInfoRow_wheel')
#     weel = element[0].text.split("\n")[1]

#     # Определяем состояние
#     element = driver.find_elements(By.CLASS_NAME,'CardInfoRow_state')
#     state = element[0].text.split("\n")[1]

#     # Определяем цену
#     element = driver.find_elements(By.CLASS_NAME,'OfferPriceCaption')
#     price = element[0].text[:-2:].replace(" ","")

#     cars.append({"brand": brand, "model_name": model_name, "bodyType":bodyType, "color":color,\
#         "engineDisplacement":engineDisplacement,'enginePower':enginePower,"fuelType":fuelType,\
#           "mileage":mileage,"productionDate":productionDate,"vehicleTransmission":vehicleTransmission,\
#           "owners":owners,"pts":pts,"wd":wd,"weel":weel,"state":state,"price":price})

#     time.sleep(1)


# In[521]:


# DIR_TRAIN  = '../input/all_auto_ru/' # подключаем удаленный датасет к ноутбуку
# DIR_TRAIN_2021  = '../input/parsed_df/' # подключаем спарсенный датасет с auto.ru
# DIR_TEST   = '../input/test_df/'


# In[522]:


VAL_SIZE= 0.20  
cols_to_remove = []


# In[523]:


get_ipython().system("ls '../input'")


# In[524]:


train = pd.read_csv('all_auto_ru.csv')  # датасет для обучения модели
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')


# In[525]:


pd.options.display.max_columns = None


# In[526]:


train.sample(3)


# In[527]:


test.sample(3)


# In[528]:


train.info()


# In[529]:


test.info()


# In[530]:


# проанализируем отличие признаков между тестовой и тренировочной выборкой
set(test.columns).difference(train.columns)


# In[531]:


set(train.columns).difference(test.columns)


# Видно, что в данных train есть 4 признака ('hidden', 'model', 'start_date', 'Комплектация') , которых нет в тестовой выборке.
# И наоборот в тестовой есть 11 признаков, которых нет в train

# ### Проанализируем эти признаки для унификации:

# #### параметры 'model_info', 'model_name', 'model'

# Как видно из таблиц, train.model содержит ту же информацию, что и в test.model_name, поэтому просто переименуем признак.
# А признак model_info дублирует model_name, удалим test.model_info.

# In[532]:


train.rename(columns={'model': 'model_name'}, inplace=True)
test.drop('model_info', axis=1, inplace=True)


# #### параметры 'complectation_dict', 'Комплектация'

# Видим, что test.complectation_dict содержит ту же информацию, что и train['Комплектация'], поэтому просто переименуем название признака.

# In[533]:


train.rename(columns={'Комплектация': 'complectation_dict'}, inplace=True)


# #### параметр 'priceCurrency'

# Поскольку признак не несет никакой информации, то его можно удалить.

# In[534]:


test.drop('priceCurrency', axis=1, inplace=True)


# #### параметр 'parsing_unixtime'

# Данный признак содержит даты в диапазоне от 19/10/20 до 26/10/20. Удалим признак, поскольку он не имеет влияния на цену на наш взгляд

# In[535]:


test.drop('parsing_unixtime', axis=1, inplace=True)


# #### параметры 'sell_id', 'price' 

# Добавим нулевые значения в train и test

# In[536]:


test['price'] = 0
train['sell_id'] = 0


# Удалим оставшиеся признаки, которые есть только в одной из выборок, поскольку они не имееют влияния на цену:

# In[537]:


test.drop(['car_url', 'equipment_dict', 'image',
           'super_gen', 'vendor'], axis=1, inplace=True)
train.drop(['hidden', 'start_date'], axis=1, inplace=True)


# In[538]:


test.info()


# Признаки 'Владение'  и 'complectation_dict' имеют большое количество пропусков. Удалим эти столбцы                            

# In[539]:


test.drop(['complectation_dict', 'Владение'], axis=1, inplace=True)
train.drop(['complectation_dict', 'Владение'], axis=1, inplace=True)


# Рассмотрим, сколько брендов представлено в нашей выборке:

# In[540]:


train.brand.sort_values().unique(), test.brand.sort_values().unique()


# Поскольку в тестовой выборке всего 12 брендов, уберем лишние бренды из train

# In[541]:


train = train[train.brand.isin(test.brand.unique())]


# Признак 'Состояние' содержит только 1 значение, поэтому его можно удалить

# In[542]:


test.drop('Состояние', axis=1, inplace=True)
train.drop('Состояние', axis=1, inplace=True)


# ### Работа с пропусками

# In[543]:


train.info()


# In[544]:


# пропуски в графе 'Владельцы' можно заменить на наиболее часто повторяющееся значение
train['Владельцы'].value_counts()


# In[545]:


train['Владельцы'].fillna(3, inplace=True)


# In[546]:


# удалим признак description
test.drop('description', axis=1, inplace=True)
train.drop('description', axis=1, inplace=True)


# In[547]:


train.dropna(subset=['price'], inplace=True)


# In[548]:


# заменим пропуски в 'ПТС' на 'Оригинал'
train['ПТС'].fillna('Оригинал', inplace=True)


# In[549]:


train.dropna(inplace=True)


# In[550]:


train.info()


# In[551]:


test.info()


# Добавим в выборку файл с собранными данными

# In[552]:


train_2021 = pd.read_csv('parsed_df.csv')
train_2021.sample(3)


# Найдем отличия с тестовой таблицей

# In[553]:


set(test.columns).difference(train_2021.columns)


# In[554]:


pars = train_2021.copy()
pars.drop(['car_url', 'equipment_dict', 'image', 'super_gen', 'priceCurrency', 'parsing_unixtime', 'complectation_dict',
           'Владение', 'Состояние', 'description', 'views', 'date_added',  'region'], axis=1, inplace=True)


# In[555]:


pars.dropna(subset=['price'], inplace=True)


# In[556]:


pars.dropna(subset=['fuelType'], inplace=True)


# In[557]:


pars.dropna(inplace=True)


# In[558]:


pars.info()


# In[559]:


# удалим повторы
set(pars.columns).difference(test.columns)


# # Предварительная обработка данных и EDA

# ### Объединим train и test датасеты

# In[560]:


# добавим столбец, который будет показывать train или test, чтобы было легче их объединять и разделять
train['train'] = 1
pars['train'] = 1
test['train'] = 0

# эту колонку следует добавить к train, потому что она есть в test, и используется для submission
train['sell_id'] = 0

#train_2021['sell_id'] = 0
pars['sell_id'] = 0


# In[561]:


combined_df = pd.concat([test, train, pars], join='inner', ignore_index=True)
print(combined_df.shape)


# In[562]:


test.columns


# ### Построим наивную модель

# Сначала изменим категориальные признаки

# In[563]:


for col in ['brand', 'model_name']:
    combined_df[col] = combined_df[col].astype('category').cat.codes


# In[564]:


naiv = combined_df.columns[combined_df.dtypes != object]


# In[565]:


naiv = combined_df[naiv]


# In[566]:


naiv['age'] = 2021-naiv['productionDate']


# In[567]:


naiv[naiv.sell_id == 0]


# In[568]:


X = naiv[naiv.sell_id == 0].drop(['price', 'train', 'productionDate'], axis=1)
y = naiv[naiv.sell_id == 0].price

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=VAL_SIZE,  random_state=RANDOM_SEED)

lr = LinearRegression().fit(X_train, y_train)
y_pred = (lr.predict(X_test))

print(
    f"The accuracy of the naive model using MAPE metrics is : {(mape(y_test, y_pred))*100:0.2f}%.")


# Наивная модель дала результат 130. С ним будем сравнивать результаты других моделей. 

# #### Рассмотрим каждый признак и, по возможности, заменим категориальный признак на числовой или бинарный

# In[569]:


combined_df.sample(3)


# In[570]:


combined_df.bodyType.unique()


# In[571]:


combined_df.bodyType = combined_df.bodyType.apply(
    lambda x: x.lower().split()[0].strip() if isinstance(x, str) else x)


# In[572]:


combined_df.bodyType.unique()


# In[573]:


combined_df.bodyType = combined_df['bodyType'].astype('category').cat.codes


# #### признак "цвет"

# In[574]:


combined_df.color.value_counts()


# In[575]:


# используем словарь для перевода кодировки цветов:
color_dict = {'040001': 'чёрный', 'FAFBFB': 'белый', '97948F': 'серый', 'CACECB': 'серебристый', '0000CC': 'синий', '200204': 'коричневый',
              'EE1D19': 'красный',  '007F00': 'зелёный', 'C49648': 'бежевый', '22A0F8': 'голубой', '660099': 'пурпурный', 'DEA522': 'золотистый',
              '4A2197': 'фиолетовый', 'FFD600': 'жёлтый', 'FF8649': 'оранжевый', 'FFC0CB': 'розовый'}
combined_df.color.replace(color_dict, inplace=True)
combined_df.color.value_counts(normalize=True)


# In[576]:


for col in ['color']:
    combined_df[col] = combined_df[col].astype('category').cat.codes


# #### признак "рабочий объем"

# In[577]:


combined_df.engineDisplacement = combined_df.name.apply(
    transf_engineDisplacement_to_float)


# In[578]:


combined_df.engineDisplacement.unique()


# In[579]:


test[test.brand == 'MERCEDES'].engineDisplacement.unique()


# In[580]:


combined_df[combined_df.engineDisplacement.isna()].fuelType.unique()


# In[581]:


# так как значения nan у признака "рабочий объем" есть только у электромашин, то заполним значения 0
combined_df.engineDisplacement.fillna(0, inplace=True)


# #### признак "тип топлива"

# In[582]:


combined_df.fuelType.value_counts()


# In[583]:


# разделим этот признак на две группы: бензин и не бензин
combined_df.fuelType = combined_df.fuelType.apply(
    lambda x: 1 if x == 'бензин' else 0)


# In[584]:


combined_df.fuelType.value_counts()


# #### признак "дата производства"

# In[585]:


# заменим дату на возраст
combined_df['age'] = 2021 - combined_df.productionDate


# In[586]:


combined_df['age']


# #### признак "тип коробки передач"

# In[587]:


combined_df.vehicleTransmission.value_counts()


# In[588]:


automat = ['AUTOMATIC', 'автоматическая',  'VARIATOR',
           'ROBOT', 'вариатор', 'роботизированная']
mechanic = ['MECHANICAL', 'механическая']


# In[589]:


# разделим параметры на две группы: механическая и автоматическая
combined_df.vehicleTransmission = combined_df.vehicleTransmission.apply(
    lambda x: 1 if x in automat else 0)


# In[590]:


combined_df.vehicleTransmission.value_counts()


# #### параметр "владельцы"

# In[591]:


print(combined_df['Владельцы'].unique())


# In[592]:


combined_df['owners'] = combined_df['Владельцы'].apply(transf_owners_to_float)
combined_df.owners.unique()


# #### признак "птс"

# In[593]:


print(combined_df['ПТС'].unique())


# In[594]:


combined_df['ПТС'].value_counts()


# In[595]:


combined_df.drop(['ПТС', 'Владельцы'], axis=1, inplace=True)


# #### признак "трансмиссия"

# In[596]:


print(combined_df['Привод'].unique())


# In[597]:


combined_df['Привод'].value_counts()


# In[598]:


# присвоим численные значения категориальному признаку
combined_df['transmission '] = combined_df['Привод'].apply(
    lambda x: 1 if x == 'полный' else 2 if x == 'передний' else 3)


# In[599]:


combined_df.drop(['Привод', 'Руль', 'Таможня'], axis=1, inplace=True)


# In[600]:


data_col = combined_df.columns[combined_df.dtypes != object]
data = combined_df[data_col]
data


# ### Linear Regression

# In[601]:


X = data[data.sell_id == 0].drop(
    ['price', 'train', 'productionDate'], axis=1)
y = data[data.sell_id == 0].price

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=VAL_SIZE,  random_state=RANDOM_SEED)

lr = LinearRegression().fit(X_train, y_train)
y_pred = (lr.predict(X_test))

print(
    f"The accuracy of the naive model using MAPE metrics is : {(mape(y_test, y_pred))*100:0.2f}%.")


# Линейная регрессия не показала никаких улучшений. Поработаем с выбросами и проверим результат.

# #### Работа с выбросами

# In[602]:


combined_df.sample(3)


# In[603]:


for col in data.columns:
    display(vis_num_feature(combined_df, col, 'price', 'train == 1'))
    display(calculate_stat_outliers(combined_df, col, log=True))
    print('\n' + '-' * 10 + '\n')


# Признак 'enginePower' имеет выбросы, избавимся от них.

# In[604]:


combined_df.enginePower = combined_df.enginePower.apply(
    transf_enginePower_to_float)


# In[606]:


combined_df.enginePower


# In[607]:


combined_df[combined_df.enginePower > 640]


# In[608]:


# удалим строки из train
combined_df = combined_df[combined_df.enginePower < 640]


# #### параметр "возраст машины"

# In[609]:


combined_df[combined_df['train'] == 1].age.sort_values()


# In[610]:


sd = combined_df[combined_df['train'] == 0]
sd[sd.age > 80]


# In[611]:


plt.figure(figsize=(25, 6))
sns.scatterplot(
    data=combined_df[combined_df['train'] == 1], x='age', y="price")


# #### параметр "цена"

# In[612]:


combined_df.query('train == 1').price.hist()
plt.title('The target variable distribution', fontdict={'fontsize': 14})
plt.xlabel('price ')


# Распределение имеет тяжелый левый хвост,чтобы это исправить воспользуемся логорифмированием.

# In[613]:


np.log2(combined_df.query('train == 1').price).hist()
plt.title('The log2 target variable distribution', fontdict={'fontsize': 14})


# In[614]:


# добавим новый признак
combined_df['price_log2'] = np.log2(combined_df.price + 1)


# # Feature engineering

# Создадим новые признаки:
# 
# - mileage_per_year: с помощью the productionDate и mileage columns получим информацию,сколько км проехал автомобиль за год;
# - rarity: был ли автомобиль произведен ранее 1960;
# - older_3y: старше ли автомобиль трёх лет;
# - older_5y: старше ли автомобиль пяти лет;

# In[615]:


combined_df['mileage_per_year'] = combined_df.mileage / combined_df.age
combined_df['rarity'] = combined_df.productionDate.apply(
    lambda x: 1 if x < 1960 else 0)
combined_df['older_3y'] = combined_df.age.apply(lambda x: 1 if x > 3 else 0)
combined_df['older_5y'] = combined_df.age.apply(lambda x: 1 if x > 5 else 0)


# In[616]:


combined_df.mileage_per_year = combined_df.apply(
    lambda x: x.mileage if x.age == 0 else x.mileage / x.age, axis=1)
combined_df[combined_df.age == 0]


# In[617]:


combined_df.query('train == 1').mileage_per_year.hist(bins=10)
plt.title('mileage_per_year distribution', fontdict={'fontsize': 14})


# Снова тяжелый хвост. Логорифмируем:

# In[618]:


np.log2(combined_df.query('train == 1').mileage_per_year+1).hist()
plt.title('The log2 mileage_per_year distribution', fontdict={'fontsize': 14})


# In[619]:


# добавим новый признак
combined_df['mileage_per_year_log2'] = np.log2(
    combined_df.mileage_per_year + 1)


# In[620]:


combined_df.enginePower.hist(bins=10)


# In[621]:


combined_df.columns


# In[622]:


temp1 = combined_df.copy()
temp1.sample(2)


# # Построение моделей ML

# In[623]:


X = combined_df.query('train == 1').drop(
    ['price', 'train', 'name', 'price_log2', 'mileage_per_year'], axis=1)

y = combined_df.query('train == 1').price
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)


# In[624]:


lr = LinearRegression().fit(X_train, y_train)
y_pred = (lr.predict(X_test))

print(
    f"The accuracy of the naive model using MAPE metrics is : {(mape(y_test, y_pred))*100:0.2f}%.")


# Работа с данными привела к улучшению результата модели (MAPE 91.09%), однако на практике ее применение не несёт особой пользы. Построим другие модели

# ### RandomForestRegressor

# In[625]:


rf = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbose=1)
rf.fit(X_train, y_train)
predict_rf = rf.predict(X_test)

print(
    f"The MAPE mertics of the Random Forest model using MAPE metrics: {(mape(y_test, predict_rf) * 100):0.2f}%.")

# with log-transformation of the target variable
rf_log = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbose=1)
rf_log.fit(X_train, np.log(y_train))
predict_rf_log = np.exp(rf_log.predict(X_test))

print(
    f"The MAPE mertic for the Random Forest model is : {(mape(y_test, predict_rf_log) * 100):0.2f}%.")


# Результат MAPE 14.20%

# ### ExtraTreesRegressor

# In[626]:


X = combined_df.query('train == 1').drop(
    ['price', 'train', 'name', 'price_log2', 'mileage_per_year'], axis=1)
y = combined_df.query('train == 1').price
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)


# In[627]:


X1 = np.array(X)
y1 = np.array(y)

etr_log = ExtraTreesRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbose=1)

skf = KFold(n_splits=4)
etr_log_mape_values = []

for train_index, test_index in skf.split(X1, y1):
    X_train, X_test = X1[train_index], X1[test_index]
    y_train, y_test = y1[train_index], y1[test_index]

    etr_log.fit(X_train, np.log(y_train))
    y_pred = np.exp(etr_log.predict(X_test))

    etr_log_mape_value = mape(y_test, y_pred)
    etr_log_mape_values.append(etr_log_mape_value)
    print(etr_log_mape_value)

print(
    f"The MAPE mertic for the default ExtraTreesRegressor model using 4-fold CV is: {(np.mean(etr_log_mape_values) * 100):0.2f}%.")


# В данном случае результат MAPE 22.67%

# ### XGBoostRegressor

# In[628]:


X1 = np.array(X)
y1 = np.array(y)

xgb_log = xgb.XGBRegressor(max_depth=10,
                           n_estimators=1000,
                           random_state=RANDOM_SEED,
                           n_jobs=-1)
skf = KFold(n_splits=4)
xgb_log_mape_values = []

for train_index, test_index in skf.split(X1, y1):
    X_train, X_test = X1[train_index], X1[test_index]
    y_train, y_test = y1[train_index], y1[test_index]

    xgb_log.fit(X_train, np.log(y_train))
    y_pred = np.exp(xgb_log.predict(X_test))

    xgb_log_mape_value = mape(y_test, y_pred)
    xgb_log_mape_values.append(xgb_log_mape_value)
    print(xgb_log_mape_value)

print(
    f"The MAPE mertic for the XGBRegressor model using 4-fold CV: {(np.mean(xgb_log_mape_values) * 100):0.2f}%.")


# Здесь метрика MAPE 22.25%

# ### StackingRegressor

# In[629]:


estimators = [
    ('etr', ExtraTreesRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbose=1)),
    ('xgb', xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.5, learning_rate=0.05,
                             max_depth=12, alpha=1, n_jobs=-1, n_estimators=1000, random_state=RANDOM_SEED))]

sr_log = StackingRegressor(estimators=estimators,
                           final_estimator=LinearRegression())

sr_log.fit(X_train, np.log(y_train))

y_pred = np.exp(sr_log.predict(X_test))

print(
    f"The MAPE mertic for the default StackingRegressor model: {(mape(y_test, y_pred) * 100):0.2f}%.")


# MAPE 19.17%

# # Выводы по моделям:
# 
# Лучший результат метрики MAPE показала модель RandomForestRegressor(14.20%), а худший - ExtraTreesRegressor с результатом 22.67%. Средние результаты у моделей XGBoostRegressor и StackingRegressor (22.25% и 19.17% соответственно). Предположительно, высокий показатель метрики у RandomForestRegressor стал результатом переобучения модели. Поэтому в качестве модели для final submission был выбран стэкинг.

# # Submission

# In[636]:


X_kag = combined_df.query('train == 0').drop(
    ['price', 'train', 'name', 'price_log2', 'mileage_per_year'], axis=1)


# In[637]:


X_kag.info()


# In[638]:


X_kag.enginePower = X_kag.enginePower.apply(
    transf_engineDisplacement_to_float)


# In[640]:


predict_submission = np.exp(sr_log.predict(X_kag))
sample_submission['price'] = predict_submission
sample_submission.to_csv(f'submission_final.csv', index=False)
sample_submission.head(10)


# In[ ]:




