#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind

pd.set_option('display.max_rows', 50)  # показывать больше строк
pd.set_option('display.max_columns', 50)  # показывать больше колонок

stud = pd.read_csv('stud_math.csv')


# In[96]:


display(stud.head(10))
stud.info()


# В датасете всего 13 числовых столбцов и 17 строковых. 
# Видим, что один столбец имеет название, которое не будет читаться. Заменим запятую и пробел на нижнее подчеркивание

# In[97]:


stud.columns = stud.columns.str.replace(', ', '_')


# In[98]:


stud.columns


# По информации о датасете можно понять, что пустых значений не очень много. В среднем не более 10%.
# Определим функцию замены отсутствующих значений на медианные

# In[99]:


def replacemed(x):
    x = x.fillna(x.median())
    return x


# Для начала проверим все численные столбцы на наличие выбросов и количество пустых значений

# In[100]:


print("Количество пустых значений:", (stud.age.isna().sum()))
pd.DataFrame(stud.age.value_counts())


# Посмотрим на распределение данных в столбце 

# In[101]:


stud['age'].plot(kind='hist', grid=True, title='Возраст')
stud.age.describe()


# Видно, что пустых значений и выбросов нет. Данные соответствуют условию, что возраст от 15 до 22 лет. 
# Основное число учеников составляют учащиеся в возрасте от 15 до 17 лет.

# In[102]:


print("Количество пустых значений:", (stud.Medu.isna().sum()))
pd.DataFrame(stud.Medu.value_counts())


# Распределение данных

# In[103]:


stud['Medu'].plot(kind='hist', grid=True, title='Образование матери')
stud.Medu.describe()


# Видим, что есть 3 нулевых значения. Заменим их на медианные значения. 
# В целом, у большинства матерей учеников высшее образование

# In[104]:


replacemed(stud.Medu)


# In[105]:


print("Количество пустых значений:", (stud.Fedu.isna().sum()))
pd.DataFrame(stud.Fedu.value_counts())


# Распределение данных

# In[106]:


stud['Fedu'].plot(kind='hist', grid=True, title='Образование отца')
stud.Fedu.describe()


# Очевидно, что в данных есть выбросы. Чтобы их отфильтровать, воспользуемся формулой межквартильного размаха.

# In[107]:


IQR = stud.Fedu.quantile(0.75) - stud.Fedu.quantile(0.25)
perc25 = stud.Fedu.quantile(0.25)
perc75 = stud.Fedu.quantile(0.75)

print(
    '25-й перцентиль: {},'.format(perc25),
    '75-й перцентиль: {},'.format(perc75),
    "IQR: {}, ".format(IQR),
    "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))


# Поскольку нам дано, что значения этого столбца могут лежать в пределах от 0 до 4, то отфильтруем все остальные значения.
# Пустые ячейки можно заменить на медианы, так как их немного.

# In[108]:


replacemed(stud.Fedu)
stud = stud.loc[(stud.Fedu >= 0) & (stud.Fedu < 4.5)]


# Построим новое распределение с учетом изменений.

# In[109]:


stud['Fedu'].plot(kind='hist', grid=True, title='Образование отца')
stud.Fedu.describe()


# Здесь видно, что в большинстве случаев отцы учеников окончили 5-9 классы.

# In[110]:


print("Количество пустых значений:", (stud.traveltime.isna().sum()))
pd.DataFrame(stud.traveltime.value_counts())


# In[111]:


stud['traveltime'].plot(kind='hist', grid=True, title='Время в пути до школы')
stud.traveltime.describe()


# В этом столбце все данные находятся в требуемом диапазоне. Видно, что у более чем половины учеников время пути занимает < 15 минут.
# Есть 25 пустых значений, но так как уже сейчас можно сделать вывод о распределении данных, то заменять не будем.

# In[112]:


print("Количество пустых значений:", (stud.studytime.isna().sum()))
pd.DataFrame(stud.studytime.value_counts())


# In[113]:


stud['studytime'].plot(kind='hist', grid=True,
                       title='Время на учёбу помимо школы в неделю')
stud.studytime.describe()


# В этом столбце все данные также находятся в диапазоне от 1 до 4. У большей части учеников время время на учебу вне школы занимает 2-5 часов. 7 пустых значений заменять не будем, как и в предыдущем случае.

# In[114]:


print("Количество пустых значений:", (stud.failures.isna().sum()))
pd.DataFrame(stud.failures.value_counts())


# In[115]:


stud['failures'].plot(kind='hist', grid=True,
                      title='Количество внеучебных неудач')
stud.failures.describe()


# В данном столбце никаких аномальных значений нет. По распределению видно, что у подавляющего большинства отсутствуют внеучебные неудачи. Есть 19 пропущенных значений. Однако их замена на выводы не повлияет, поэтому оставим данные без изменений.

# In[116]:


print("Количество пустых значений:", (stud.studytime_granular.isna().sum()))
pd.DataFrame(stud.studytime_granular.value_counts())


# In[117]:


stud['studytime_granular'].plot(
    kind='hist', grid=True, title='studytime_granular')
stud.studytime_granular.describe()


# В данном столбце количество уникальных значений равно четырем, а данные разбиты на группы с численным признаком от -3 до -12 с шагом -3. Распределение значений похоже на распределение данных в столбце 'Время на учёбу помимо школы в неделю'. Поэтому будет полезным узнать насколько эти значения связаны между собой.

# In[118]:


stud['studytime'].corr(stud['studytime_granular'])


# Здесь видна отрицательная корреляция между столбцами, данные 'studytime_granular' имеют обратную зависимость от данных 'studytime', а коэффициент корреляции по модулю очень высокий. Для построения модели для переменной score достаточно оставить столбец 'studytime', а коррелирующий с ним параметр можно удалить, так как его влияение на значение score будет аналогичным.

# In[119]:


stud = stud.drop('studytime_granular', 1)


# In[120]:


print("Количество пустых значений:", (stud.famrel.isna().sum()))
pd.DataFrame(stud.famrel.value_counts())


# In[121]:


stud['famrel'].plot(kind='hist', grid=True, title='Семейные отношения')
stud.famrel.describe()


# В столбце есть значение, выходящее за пределы требуемого диапазона. Отфильтруем его, а пустые ячейки заменять не будем.

# In[122]:


stud = stud.loc[stud.famrel > (-1)]


# Судя по распределению, в семьях учеников хорошие семейные отношения.

# In[123]:


print("Количество пустых значений:", (stud.freetime.isna().sum()))
pd.DataFrame(stud.freetime.value_counts())


# In[124]:


stud['freetime'].plot(kind='hist', grid=True, title='Свободное время после школы')
stud.freetime.describe()


# Данные представлены без выбросов, не выходящие за нужный интервал. Распределение показывает среднее количество свободного времени у большинства учащихся

# In[125]:


print("Количество пустых значений:", (stud.goout.isna().sum()))
pd.DataFrame(stud.goout.value_counts())


# In[126]:


stud['goout'].plot(kind='hist', grid=True,
                   title='Проведение времени с друзьями')
stud.goout.describe()


# В данном случае также нет выбросов, данные распределены примерно равномерно, но у большинства среднее колчество времени для встреч с друзьями

# In[127]:


print("Количество пустых значений:", (stud.health.isna().sum()))
pd.DataFrame(stud.health.value_counts())


# In[128]:


stud['health'].plot(kind='hist', grid=True, title='Текущее состояние здоровья')
stud.health.describe()


# Так как в данном столбце значения распределены довольно близко, для выявления тенденции можно 13 пустых значений заменить на медианное.

# In[129]:


replacemed(stud.health)


# In[130]:


print("Количество пустых значений:", (stud.absences.isna().sum()))
pd.DataFrame(stud.absences.value_counts())


# In[131]:


stud['absences'].plot(kind='hist', grid=True,
                      title='Количество пропущенных занятий')
stud.absences.describe()


# Судя по распределению данных, в столбце есть выбросы. Найдем их, чтобы отфильтровать

# In[132]:


IQR = stud.absences.quantile(0.75) - stud.absences.quantile(0.25)
perc25 = stud.absences.quantile(0.25)
perc75 = stud.absences.quantile(0.75)

print(
    '25-й перцентиль: {},'.format(perc25),
    '75-й перцентиль: {},'.format(perc75),
    "IQR: {}, ".format(IQR),
    "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))


# Поскольку количество пропущенных занятий является неотрицательным числом, то выставим ограничения от 0 до 20, а пустые ячейки заменим медианным значением.

# In[133]:


replacemed(stud.absences)
stud = stud.loc[(stud.absences >= 0) & (stud.absences <= 20)]


# Построим новое распределение

# In[134]:


stud['absences'].plot(kind='hist', grid=True,
                      title='Количество пропущенных занятий')


# Заметна тенденция на снижение количества пропущенных занятий. Около четверти учеников вообще не пропускали занятия

# In[135]:


print("Количество пустых значений:", (stud.score.isna().sum()))
pd.DataFrame(stud.score.value_counts())


# В данном столбце есть пустые значения. Так как баллы являются целевым параметром, но пропущенные значения придется удалить 

# In[136]:


stud.score = stud.score.dropna()


# Построим распределение данных и сделаем выводы об аномальных значениях

# In[139]:


stud['score'].plot(kind='hist', grid=True,
                   title='Баллы по госэкзамену по математике')
stud.score.describe()


# Данные находятся в пределах от 0 до 100, что вполне объяснимо для баллов за экзамен. Видно, что у большинства оценки за экзамен соответствуют средним (около 52 баллов). Но также много 0 баллов, что скорее всего не является аномальным значением.

# Теперь проведем анализ номинативных переменных. Так как их довольно много, для удобства будем анализировать небольшими группами.

# In[143]:


plt.subplot(2, 2, 1)
stud['sex'].value_counts().plot(kind='bar')
plt.title("Пол")

plt.subplot(2, 2, 2)
stud['address'].value_counts().plot(kind='bar')
plt.title("Адрес")

plt.subplot(2, 2, 3)
stud['famsize'].value_counts().plot(kind='bar')
plt.title("Семья")

plt.subplot(2, 2, 4)
stud['Pstatus'].value_counts().plot(kind='bar')
plt.title("Родители")

plt.show()


# Данные по половому признаку распределены одинаково. Что касается остальных параметров, то со значительным отрывом большинство учеников живут в городе, имеют большую семью, а их родители живут вместе

# In[153]:


plt.subplot(2, 2, 1)
stud['Mjob'].value_counts().plot(kind='bar')
plt.title("Работа матери")

plt.subplot(2, 2, 2)
stud['Fjob'].value_counts().plot(kind='bar')
plt.title("Работа отца")

plt.subplot(2, 2, 3)
stud['reason'].value_counts().plot(kind='bar')
plt.title("Причина выбора школы")

plt.subplot(2, 2, 4)
stud['guardian'].value_counts().plot(kind='bar')
plt.title("Опекун")

plt.show()


# Данные по работе матерей и отцов учащихся распределены примерно одинаково. В обоих случаях большинство родителей работают в других сферах, не описанных в датасете. Причины выбора школы имеют равномерное распределение. У большинства учеников мать является опекуном

# In[145]:


plt.subplot(2, 2, 1)
stud['schoolsup'].value_counts().plot(kind='bar')
plt.title("Образовательная поддержка")

plt.subplot(2, 2, 2)
stud['famsup'].value_counts().plot(kind='bar')
plt.title("Семейная поддержка")

plt.subplot(2, 2, 3)
stud['paid'].value_counts().plot(kind='bar')
plt.title("Платные занятия")

plt.subplot(2, 2, 4)
stud['activities'].value_counts().plot(kind='bar')
plt.title("Внеучебные занятия")

plt.show()


# Большая часть учеников не получали образовательной поддержки, однако семейная поддержка преобладает. Процент дополнительных занятий распределен примерно одинаково.

# In[148]:


plt.subplot(2, 2, 1)
stud['nursery'].value_counts().plot(kind='bar')
plt.title("Детский сад")

plt.subplot(2, 2, 2)
stud['higher'].value_counts().plot(kind='bar')
plt.title("Хочет высшее образование")

plt.subplot(2, 2, 3)
stud['internet'].value_counts().plot(kind='bar')
plt.title("Интернет")

plt.subplot(2, 2, 4)
stud['romantic'].value_counts().plot(kind='bar')
plt.title("Отношения")

plt.show()


# С достаточно большой разницей преобладает количество учеников, посещавших детский сад, желающих получить высшее образование и имеющих интернет. Соотношение учеников, состоящих в отношениях отличается примерно в 2 раза.

# Построим корреляционную матрицу для числовых значений и тепловую карту для наглядности

# In[149]:


correl = stud.corr()
correl


# In[150]:


sns.heatmap(correl, cmap = 'coolwarm')


# Чтобы понять, какие параметры больше остальных влияют на итоговую оценку, отсортируем коэффициент корреляции в порядке возрастания

# In[151]:


correl.sort_values(by=['score']).loc[:, 'score']


# Видно, что отрицательная корреляция сильнее с параметром 'failures'. То есть, чем больше внеучебных неудач, тем ниже итоговая оценка. Что касается положительной корреляции, то коэффициент выше у параметра 'Medu'. Поскольку мы уже выяснили, что в большинстве семей мать является опекуном, то и образование матери имеет влияние на обучение в целом и итоговые баллы за экзамен.

# Довольно слабо коррелируют с итоговой оценкой такие параметры, как состояние здоровья, а также время, занимаемое на дорогу, свободное время и отношения в семье. Эти показатели вполне можно исключить при построении модели.

# In[ ]:




