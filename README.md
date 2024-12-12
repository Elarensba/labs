**Лаболраторная работа №2. Линейная регрессия: построение, оценка адекватности и тюнинг**

**Цель работы**
Целью работы является разработка программы для построения, оценки адекватности и тюнинга линейной многофакторной модели (ЛМФМ) на основе временных рядов данных. Программа должна позволять пользователю загружать данные, выполнять отбор значимых факторов, оценивать адекватность модели, проводить предсказание на новых данных и визуализировать результаты.

Линейная многофакторная модель (ЛМФМ) — это статистическая модель, которая используется для описания взаимосвязи между зависимой переменной (откликом) и несколькими независимыми переменными (факторами). В данной работе ЛМФМ строится на основе временных рядов данных, а её адекватность оценивается с использованием статистических критериев и метрик качества.

**1. Построение ЛМФМ**
**1.1. Загрузка и подготовка данных**
Загрузка данных:
Данные загружаются из текстового файла, содержащего временные ряды отклика (y) и факторов (X).
Данные могут быть представлены в формате Excel или CSV.
Предобработка данных:
Удаление пропусков: данные с отсутствующими значениями удаляются или заполняются (например, средним значением).
Нормализация данных: факторы могут быть нормализованы для улучшения стабильности модели (например, с использованием StandardScaler или MinMaxScaler).
Удаление выбросов: выбросы могут быть идентифицированы и удалены для улучшения качества модели.

**1.2. Построение модели**

Формула модели:
![image](https://github.com/user-attachments/assets/d421d5d8-1893-458c-8330-6963f2a951bb)


Метод оценки параметров:

Для оценки коэффициентов модели используется метод наименьших квадратов (OLS).

В библиотеке statsmodels это реализуется с помощью функции OLS (Ordinary Least Squares).

Добавление константы:

Для учета свободного члена в модель добавляется константа с помощью метода sm.add_constant(X).

**2. Отбор значимых факторов**
**2.1. Отсев факторов по статистической значимости**
Критерий p-value:

Факторы, для которых p-value превышает уровень значимости (например, 0.05), считаются незначимыми и исключаются из модели.

Пользователь может выбрать уровень значимости.

Повторный отсев:

После исключения незначимых факторов модель перестраивается, и процедура отсева повторяется до тех пор, пока все оставшиеся факторы не будут статистически значимы.

**2.2. Отсев факторов по коэффициенту корреляции**
Корреляция между факторами и откликом:

Факторы с низкой корреляцией с откликом (например, 
∣
r
∣
<
0.3
∣r∣<0.3) могут быть исключены из модели.

Корреляция между факторами:

Для устранения мультиколлинеарности (высокой корреляции между факторами) можно исключить факторы с высокой взаимной корреляцией (например, 
∣
r
∣
>
0.8
∣r∣>0.8).

**2.3. Оценка целесообразности исключения факторов**
Пользователь может принять решение об исключении факторов на основе анализа корреляций и p-value.

**3. Оценка адекватности модели**
**3.1. Коэффициент детерминации (R²)**
Формула:
![image](https://github.com/user-attachments/assets/4502f540-9e50-4369-8364-4e95ccb21ec7)



Проверяется с использованием F-статистики.

Если p-value для F-статистики меньше уровня значимости (например, 0.05), модель считается адекватной.

**3.2. Среднеквадратичная ошибка (RMSE)**
Формула:
![image](https://github.com/user-attachments/assets/5e24538e-3e37-4186-95c9-ea302afbea3b)

Интерпретация:

Чем меньше RMSE, тем лучше модель предсказывает значения отклика.

**3.3. Средняя относительная ошибка (E)**
Формула:
![image](https://github.com/user-attachments/assets/e4141538-43f5-4487-ae6c-035055a379ca)


Интерпретация:

Показывает среднюю относительную ошибку предсказания.

**4. Тюнинг модели**
**4.1. Введение временных лагов**


![image](https://github.com/user-attachments/assets/741ab7a0-9ed6-4bb0-bdff-7103a565a112)


Пользователь может выбрать количество лагов для каждого фактора.

**4.2. Выделение подряда отклика**
Отклик может быть ограничен подрядом:
![image](https://github.com/user-attachments/assets/5e9b0bce-f3fe-43c4-b9c4-fcf0a4b6e108)


Это позволяет уменьшить объем данных, сохраняя при этом достаточное количество наблюдений для построения модели.

**4.3. Оптимизация параметров**
Для минимизации ошибки (RMSE или E) можно варьировать количество лагов и подряд отклика.

Оптимальные параметры выбираются на основе анализа метрик качества.

**5. Предсказание**
На основе построенной модели выполняется предсказание отклика для новых значений факторов.

Пользователь может ввести новые значения факторов вручную или загрузить их из файла.

**6. Визуализация**
Графики фактических и предсказанных значений отклика на обучающей и тестовой выборках.

Графики ошибок модели.



**Заключение**
Процедура построения и оценки ЛМФМ включает в себя загрузку и подготовку данных, построение модели, отбор значимых факторов, оценку адекватности, тюнинг параметров и визуализацию результатов. Основная цель — построить адекватную модель, которая позволяет точно предсказывать значения отклика на основе выбранных факторов.



**Визуализация интерфейса получившейся модели**

![image](https://github.com/user-attachments/assets/1a12182f-b3f8-4c48-98cf-a3e4907b8c82)

![image](https://github.com/user-attachments/assets/4dee6548-209b-430e-92e8-8cec99b58d69)
![image](https://github.com/user-attachments/assets/f3be8eb0-b6ca-4d55-8d8e-e73c5740b786)
![image](https://github.com/user-attachments/assets/ff784650-91f8-435b-a694-1eac8a7bfcf7)
![image](https://github.com/user-attachments/assets/74822891-3ca2-4053-acb8-621fa92c40b6)


![image](https://github.com/user-attachments/assets/2deae39c-b07c-44d4-9c9b-0ac9215910b7)

![image](https://github.com/user-attachments/assets/62f0317a-dc82-4650-b0e7-7347460e4770)

![image](https://github.com/user-attachments/assets/5c773498-feb5-449a-ace7-7b61c86ce802)







