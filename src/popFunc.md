Содержание:
1. [Numpy](#numpy)  
1.1. [Numpy.Random](#numpyrandom)  
2. [Pandas](#pandas)  
3. [matplotlib](#matplotlib)  
3.1. [Matplotlib.Pyplot](#matplotlibpyplot)  
4. [seaborn](#seaborn)  
5. [scikit-learn](#scikit-learn)  
5.1. [scikit-learn.preprocessing.OneHotEncoder](#scikit-learnpreprocessingonehotencoder)  
5.2. [scikit-learn.preprocessing.LabelEncoder](#scikit-learnpreprocessinglabelencoder)  

## Numpy  
(ver 2.1)  
[**numpy.arange**](https://numpy.org/doc/stable/reference/generated/numpy.arange.html#numpy-arange) - Возвращает равномерно распределенные значения в пределах заданного интервала.  
[**numpy.dot**](https://numpy.org/doc/stable/reference/generated/numpy.dot.html#numpy.dot) - умножение матриц  
[**numpy.clip**](https://numpy.org/doc/stable/reference/generated/numpy.clip.html#numpy.clip) - Обрезайте значения в массиве. При заданном интервале значения за пределами интервала обрезаются до краев интервала. Например, если задан интервал [0, 1], значения, меньшие 0, становятся равными 0, а значения, большие 1, становятся равны 1.  
[**numpy.log**](https://numpy.org/doc/stable/reference/generated/numpy.log.html#numpy.log) - Натуральный логарифм по элементам.  
[**numpy.corrcoef**](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html#numpy-corrcoef) - Верните коэффициенты корреляции Пирсона.

### Numpy.Random  
[**numpy.random.choice**](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html#numpy.random.choice) - Генерирует случайную выборку из заданного одномерного массива.  
[**numpy.random.permutation**](https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html#numpy.random.permutation) - Произвольно переставляет последовательность или возвращает измененный диапазон.  
[**numpy.random.randint**](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html#numpy.random.randint) - Возвращает случайные целые числа от наименьшего (включительно) до наибольшего (исключая).  


## Pandas  
(ver 2.2)  
[**pandas.read_csv**](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html#pandas.read_csv) - Считать файл значений, разделенных запятыми (csv), в DataFrame.  
[**pandas.crosstab**](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html#pandas.crosstab) - Вычислите простую перекрестную таблицу двух (или более) факторов.  
[**pandas.Series**](https://pandas.pydata.org/docs/reference/api/pandas.Series.html#pandas.Series) - Одномерный массив с метками осей (включая временные ряды).  
[**pandas.DataFrame**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame) - Двумерные, изменяемые по размеру, потенциально неоднородные табличные данные.  
[**pandas.DataFrame.info**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html#pandas.DataFrame.info) - Распечатать краткое описание DataFrame.  
[**pandas.DataFrame.corr**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html#pandas.DataFrame.corr) - Вычислить попарную корреляцию столбцов, исключая значения NA/null.  
[**pandas.DataFrame.describe**]() - Создание описательной статистики. Описательная статистика включает в себя статистику, обобщающую центральную тенденцию, дисперсию и форму распределения набора данных, за исключением значений NaN.  
[**pandas.DataFrame.head**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html#pandas.DataFrame.head) - Верните первые n строк.  
[**pandas.DataFrame.sample**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html#pandas.DataFrame.sample) - Возвращает случайную выборку элементов из оси объекта.  
[**pandas.DataFrame.tail**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.tail.html#pandas.DataFrame.tail) - Верните последние n строк.  
[**pandas.DataFrame.size**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.size.html) - Возвращает целое число, представляющее количество элементов в этом объекте.  
[**pandas.DataFrame.astype**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html) - Привести объект pandas к указанному dtype dtype.  
[**pandas.DataFrame.shape**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html) - Возвращает кортеж, представляющий размерность DataFrame.  
[**pandas.DataFrame.isna**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html#pandas.DataFrame.isna) - Обнаружение пропущенных значений.  
[**pandas.DataFrame.columns**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html#pandas.DataFrame.columns) - Метки столбцов DataFrame.  
[**pandas.Series.str.extract**](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.extract.html#pandas.Series.str.extract) - Извлечь группы захвата в шаблоне регулярного выражения в виде столбцов в DataFrame.  
[**pandas.DataFrame.loc**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html#pandas.DataFrame.loc) - Доступ к группе строк и столбцов по меткам или логическому массиву.  
[**pandas.DataFrame.iloc**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html#pandas.DataFrame.iloc) - Индексация на основе чисто целочисленного местоположения для выбора по позиции. `.iloc[]` is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array.  
[**pandas.DataFrame.value_counts**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html#pandas.DataFrame.value_counts) - Возвращает ряд, содержащий частоту каждой отдельной строки в Dataframe.  
[**pandas.DataFrame.mean**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mean.html#pandas.DataFrame.mean) - Возвращает среднее значение значений по запрошенной оси.  
[**pandas.DataFrame.sum**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sum.html) - Возвращает сумму значений по запрошенной оси.  
[**pandas.DataFrame.drop**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html#pandas.DataFrame.drop) - Удалить указанные метки из строк или столбцов.  
[**pandas.DataFrame.dropna**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html#pandas.DataFrame.dropna) - Удалить пропущенные значения.  
[**pandas.DataFrame.quantile**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.quantile.html#pandas.DataFrame.quantile) - Возвращает значения в заданном квантиле по запрошенной оси.  
[**pandas.DataFrame.replace**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html#pandas.DataFrame.replace) - Заменить значения, указанные в to_replace, на значение  
[**pandas.DataFrame.groupby**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html#pandas.DataFrame.groupby) - Группируйте DataFrame с помощью картографа или по серии столбцов.  
[**pandas.DataFrame.select_dtypes**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html) - Возвращает подмножество столбцов DataFrame на основе dtypes столбцов.  
[**pandas.concat**](https://pandas.pydata.org/docs/reference/api/pandas.concat.html#pandas.concat) - Объединить объекты pandas вдоль определенной оси.  
[**pandas.get_dummies**](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html#pandas.get_dummies) - Преобразовать категориальную переменную в фиктивные/индикационные переменные.  
[**pandas.DataFrame.nlargest**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nlargest.html) - Верните первые n строк, упорядоченных по столбцам в порядке убывания.  



## Matplotlib  
(ver. 3.4.3)  
### Matplotlib.Pyplot
[**matplotlib.pyplot.figure**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.figure.html?highlight=figure#matplotlib.pyplot.figure) - Создайте новый график или активируйте существующий.  
[**matplotlib.pyplot.plot**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.plot.html?highlight=plot#matplotlib.pyplot.plot) - Постройте график зависимости y от x в виде линий и/или маркеров.  
[**matplotlib.pyplot.subplots**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.subplots.html?highlight=subplots#matplotlib.pyplot.subplots) - Создание нескольких графиков на одном рисунке.  
[**matplotlib.pyplot.bar**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.bar.html?highlight=bar#matplotlib.pyplot.bar) - Построить столбчатую диаграмму.  
[**matplotlib.pyplot.barh**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.barh.html?highlight=barh#matplotlib.pyplot.barh) - Построить горизонтальную столбчатую диаграмму.  
[**matplotlib.pyplot.pie**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.pie.html?highlight=pie#matplotlib.pyplot.pie) - Постройте круговую диаграмму.  
[**matplotlib.pyplot.boxplot**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.boxplot.html?highlight=boxplot#matplotlib.pyplot.boxplot) - Постройте диаграмму боксплот.  
[**matplotlib.pyplot.axvline**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.axvline.html?highlight=axvline#matplotlib.pyplot.axvline) - Добавьте вертикальную линию по осям.  
[**matplotlib.pyplot.title**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.title.html?highlight=title#matplotlib.pyplot.title) - даёт название заготовку.  
[**matplotlib.pyplot.xlabel**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.xlabel.html?highlight=xlabel#matplotlib.pyplot.xlabel) - Добавить подпись для оси X.  
[**matplotlib.pyplot.ylabel**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.ylabel.html?highlight=ylabel#matplotlib.pyplot.ylabel) - Добавить подпись для оси Y.  
[**matplotlib.pyplot.xlim**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.xlim.html?highlight=xlim#matplotlib.pyplot.xlim) - Получить или установить пределы x текущих осей.  
[**matplotlib.pyplot.ylim**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.ylim.html?highlight=ylim#matplotlib.pyplot.ylim) - Получить или установить пределы Y для текущих осей.  
[**matplotlib.pyplot.show**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.show.html?highlight=show#matplotlib.pyplot.show) - Показать все открытые графики.  
[**matplotlib.pyplot.suptitle**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.suptitle.html?highlight=suptitle#matplotlib.pyplot.suptitle) - Добавьте к рисунку центральный заголовок.  
[**matplotlib.pyplot.scatter**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.scatter.html?highlight=scatter#matplotlib.pyplot.scatter) - Диаграмма точек y и x с различным размером и/или цветом маркера.  
[**matplotlib.pyplot.grid**](https://matplotlib.org/3.4.3/api/_as_gen/matplotlib.pyplot.grid.html?highlight=grid#matplotlib.pyplot.grid) - Отрисовать линии сетки.  


## Seaborn
(ver. 0.13.2)  
[**seaborn.heatmap**](https://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap) - Представить прямоугольные данные в виде матрицы с цветовой кодировкой.  
[**seaborn.histplot**](https://seaborn.pydata.org/generated/seaborn.histplot.html#seaborn-histplot) - Постройте одномерные или двумерные гистограммы для отображения распределений наборов данных.  
[**seaborn.pairplot**](https://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn-pairplot) - Постройте парные отношения в наборе данных.  


## scikit-learn
(ver. 1.5.2)

### scikit-learn.preprocessing.OneHotEncoder

[**fit_transform**](https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder.fit_transform) - Подгоните данные под себя, а затем преобразуйте их.  

### scikit-learn.preprocessing.LabelEncoder

[**fit_transform**](https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder.fit_transform) - Установите кодировщик меток и верните закодированные метки.  

### sklearn.model_selection

[**train_test_split**](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) - Разбить массивы или матрицы на случайные обучающие и тестовые подмножества.

### sklearn.metrics

[**root_mean_squared_error**](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.root_mean_squared_error.html) - Среднеквадратическая ошибка регрессионных потерь.


## Метрики качества  
**Accuracy** (Точность) - это метрика, которая измеряет, насколько точно модель предсказывает правильный ответ. Она рассчитывается как количество правильных ответов, полученных от модели, деленное на общее количество предсказаний.  

**Precision** (Точность) - это метрика, которая показывает, насколько точно модель идентифицирует положительные примеры. Она рассчитывается как количество истинно положительных ответов, деленное на общее количество предсказанных положительных ответов.

**Recall** (Полнота) - это метрика, которая показывает, насколько хорошо модель находит все положительные примеры. Она рассчитывается как количество истинно положительных ответов, деленное на общее количество истинных положительных ответов.

**F1 score** - это метрика, которая объединяет точность и полноту, показывая, как хорошо модель предсказывает положительные примеры. Она рассчитывается как гармоническое среднее между точностью и полнотой.

**Mean Squared Error (MSE)** - это метрика, которая измеряет среднеквадратичную ошибку модели. Она рассчитывается как сумма квадратов разности между фактическим и предсказанным значением, деленной на общее количество примеров.

**Mean Absolute Error (MAE)** - это метрика, которая измеряет среднюю абсолютную ошибку модели. Она рассчитывается как сумма абсолютных разностей между фактическим и предсказанным значением, деленной на общее количество примеров.

**R2 score** - это метрика, которая измеряет долю дисперсии в целевой переменной, которую модель может объяснить. Она рассчитывается как коэффициент детерминации между фактическим и предсказанным значением.

**AUC-ROC** - это метрика, которая измеряет площадь под кривой ROC (Receiver Operating Characteristic), которая показывает, насколько хорошо модель различает между классами. Чем ближе AUC-ROC к 1, тем лучше модель различает между классами.

**Log Loss** - это метрика, которая измеряет ошибку логарифма вероятности, которую модель предсказывает для каждого класса. Она широко используется в задачах классификации, особенно в задачах с несбалансированными классами.

**Mean Average Precision (MAP)** - это метрика, которая измеряет среднюю точность модели в ранжировании результатов по релевантности. Она широко используется в задачах информационного поиска, например, в поисковых системах.

**Cohen's Kappa** - это метрика, которая измеряет степень согласованности между двумя аннотаторами или между аннотатором и моделью. Она часто используется в задачах обработки естественного языка для оценки качества аннотации.

**IoU (Intersection over Union)** - это метрика, которая используется в задачах сегментации изображений и оценивает насколько хорошо модель выделяет объекты на изображении. Она рассчитывается как отношение пересечения и объединения между предсказанной и фактической маской.

**BLEU (Bilingual Evaluation Understudy)** - это метрика, которая измеряет качество машинного перевода, сравнивая перевод модели с несколькими референсными переводами. Она широко используется в задачах машинного перевода.

**R-squared** - это метрика, которая используется в задачах регрессии для оценки того, насколько хорошо модель соответствует реальным данным. Она измеряет долю вариации в целевой переменной, которая объясняется моделью.

**Precision-Recall Curve** - это кривая, которая показывает зависимость между точностью и полнотой модели в задачах бинарной классификации. Она помогает выбрать наилучший порог для классификации, которая максимизирует F1-score.

**Top-k Accuracy** - это метрика, которая измеряет долю правильных ответов модели, когда рассматриваются только k наиболее вероятных классов. Она используется в задачах с большим количеством классов, чтобы измерить качество модели, когда невозможно рассмотреть все классы.



