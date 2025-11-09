import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from setuptools.archive_util import default_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from statsmodels.graphics.mosaicplot import mosaic


# Загружаем файл
url="https://drive.google.com/uc?id=1t0HI2RKZAAjCvvkQMybCZEd8xsY6Xxii"
df=pd.read_csv(url, delimiter = ",")


#Смотрим информацию по таблице
print(df.info())
print(f"Размер данных: {df.shape}")
print(df.dtypes)
print(df.describe())

#Анализируем и удаляем дубликаты
print(f"Количество дубликатов: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# Проверяем пропуски и визуализируем пропуски
print(df.isnull().sum())
print(df.isna().sum()) # делает то же, что и предыдущая функция

plt.figure(figsize(10,6))
sns.heatmap(df.isna(), cbar=False)
plt.title("Визуализация пропусков в колонках")
plt.xlabel("Колонки", fontsize = 12)
plt.ylabel("Пропуски")
#plt.grid(True, alpha = 0.3)
#plt.xticks(rotation = 45)
plt.show()

# Удаляем выбросы по возрасту

print(df["Age"].describe())

a = df["Age"].quantile(0.25)
b = df['Age'].quantile(0.75)
df = df[(df["Age"] < b + 1.5* (b-a)) & (df["Age"]> a - 1.5*(b-a))]

sns.boxplot(df["Age"])
plt.show()

print(df["Age"].describe())

# Вставляем значения в пропуски

df_dropped = df.dropna(how="all") # Удаление только строк, где все значения — пропуски
df["Weight"] = df["Weight"].fillna(df["Weight"].mean())
df['Removed teeth'] = df['Removed teeth'].fillna(df['Removed teeth'].mean())
df["Children number"] = df["Children number"].fillna(df["Children number"].mode()[0])
df["Glasses"] = df["Glasses"].fillna("нет")
print(f"Пропуски в обработанных столбцах: {df.isnull().sum()}")

# Min-Max нормализация:
scaler = MinMaxScaler()
df[["City population"]] = scaler.fit_transform(df[["City population"]])
print("После масштабирования:")
print(df['City population'].head(10))

plt.figure(figsize=(8,4))
plt.hist(df['City population'], bins=20, color='dodgerblue', edgecolor='black')
plt.title('Распределение масштабированной численности города (City population)')
plt.xlabel('Масштабированное значение (0 — самый маленький, 1 — самый большой)')
plt.ylabel('Число студентов')
plt.grid(axis='y', alpha=0.3)
plt.show()


# Вытаскиваем определенную строчку
print(df.loc[55])

# Группируем данные из таблицы и строим столбчатую диаграмму
fem = df[df["Sex"]=="женский"]
male = df[df["Sex"]=="мужской"]
print(fem["Coin"].value_counts(), male["Coin"].value_counts())

sns.countplot(data = df, x = "Sex", hue = "Coin")
plt.show()

# Группируем по полу и росту и выводим среднее значение для каждого пола
print(df.groupby("Sex")["Growth"].mean())
print(f"Группировка детей:{df.groupby("Sex")["Children number"].sum()}")

# Преобразуем данные в другой формат

df["Month of birthday"] = df["Month of birthday"].astype("str")
print(df["Month of birthday"].dtype)

# Преобразуем категориальные переменные
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
print(df["Sex"].head(10))
df["Sex"] = le.inverse_transform(df["Sex"])

# Строим гистограмму
a = df["Friend number"].quantile(0.25)
b = df["Friend number"].quantile(0.75)
df = df[(df["Friend number"]<b+1.5*(b-a))&(df["Friend number"]>a - 1.5*(b-a))]
print(df["Friend number"].describe())
sns.histplot(df["Friend number"], kde=True)
plt.show()

# Смотрим корреляции
df_num = df.select_dtypes(include="number")
corr_matrix_pearson  = df_num.corr(method="pearson")
corr_matrix_spear = df_num.corr(method="spearman")
sns.heatmap(corr_matrix_pearson)

plt.show()
sns.heatmap(corr_matrix_spear)

plt.show()



plt.show()

fig, axes = plt.subplots(1, 2, figsize= (10,10))

sns.scatterplot(x = "Weight", y = "Growth", data = df, ax=axes[0])
axes[0].set_title("Вес и рост")

sns.scatterplot(x = "Maths rating", y = "Russian rating", hue = "Sex", data = df, ax=axes[1])
axes[1].set_title("Баллы по математике и русскому")

plt.tight_layout()
plt.show()


value_counts = df["Brother-sister"].value_counts()
cat = value_counts.index
counts = value_counts.values
plt.bar(cat, counts)
plt.title("Распределение братьев и сестер")
plt.xlabel("Наличие братьев и сестер")
plt.ylabel("Количество респондентов")
plt.show()

plt.pie(counts, labels = cat,  autopct='%1.1f%%')
plt.title("Доли сиблингов")
plt.show()

value_counts = df["Rock paper scissors"].value_counts()
cat1 = value_counts.index
counts1 = value_counts.values
plt.bar(cat1, counts1)
plt.title("Камень, ножницы, бумага, ящерица, Спок")
plt.xlabel("Выбор")
plt.ylabel("Количество респондентов")
plt.show()

plt.pie(counts1, labels = cat1,  autopct='%1.1f%%')
plt.title("Доли")
plt.show()


mosaic(df,["Sex", "Animal"])
plt.show()


sns.boxplot(data=df, x = "Chocolate", y = "Russian rating")
plt.show()

print(pd.crosstab(df['Chocolate'], df["Sex"], normalize="index"))

pivot = pd.pivot_table(
    df,
    values="Growth",
    index="Sex",
    aggfunc="mean"
)
print(pivot)

pivot2 = pd.pivot_table(
    df,
    values='Maths rating',
    index='Sex',
    columns='Hostel',
    aggfunc='mean'
)
print(pivot2)

# Фильтрация данных

df_filtered = df[(df["Growth"] >= df["Growth"].mean()) & (df["Weight"]>= df["Weight"].mean())]
print(df_filtered.info())

df_choc = df.query("Chocolate == 'КитКат'")
print(df_choc.info())

print(df["Fastfood"].value_counts())
df_ff = df[df["Fastfood"].isin(["KFC","Макдональдс (или как он там сейчас называется?)"])]
print(df_ff.info())


# Выгружаем обработанную таблицу
df.to_csv('cleaned_data.csv', index=False)