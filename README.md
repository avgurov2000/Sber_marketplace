<h2 align="center">Защита визуального контента маркетплейсов путем стеганографии в цифровых изображениях</h2>
<h5 align="center">НЦКР ITMO, Санкт-Петербург, Россия</h5>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


## Содержание



## Аугментации

В качестве задачи стоит разработка метода встраивания цифровых водяных знаков в изображения товаров маркетплейсов, наибольший интерес вызывает устойчивость такого метода защиты к геометрическим аугментациям защищенных изображений. В связи с чем нами был определен набор аугментация/шума, относительно которого будет проводиться обучение и тестирование моделей. Набор аугментаций и их параметров перечислен ниже в таблице:

| Аугментация       | Описание                                        | Параметры        |
| -------------     |-------------                                    | -------------    |
| *Identity*         | В этом случае изображение с встроенной меткой никак не изменяется. Использование не измененного изображения в качестве атаки при обучение призвано  предотвратить деградацию точности при извлечения водяного знака.   | |
| *Crop*              | Данный вид атаки обрезает изображение с защитной меткой до части меньшего размера. | Соотношение размеров исходного и обрезанного изображения, выбирается случайно из диапазона [0.25, 1].|
| *Cropout*           | Данный вид атаки используется для комбинирования части изображения с защищенной меткой и оставшейся части без него. | Соотношение размера исходного изображения и размера части изображения со встроенной меткой, выбирается случайно из диапазона [0.25, 1]. |
| *Dropout*           | Данный вид атаки используется для комбинирования части изображения с защищенной меткой и оставшейся части без него. В отличии от атаки Cropout, часть изображения с защищенной меткой в этом случае представляет собой не определенный вырезанный прямоугольный кусок защищенного изображения, а случайно выбранные пиксели такого изображения. Соответственно, оставшиеся пиксели представляют собой пиксели незащищенного изображения. | Соотношение пикселей защищенного изображения к количеству всех пикселей изображения, выбирается случайно из диапазона [0.75, 1].| 
| *Jpeg*              | При данной атаки к исходному защищенному изображению применятся jpeg-компрессия. |  | 
| *Rotate*            | Данная атака используется для поворота защищенного изображения на определенный угол. | Угол поворота в градусах, выбирается случайно из диапазона [-15, 15]. |
| *Hflip*             | Данная атака используется для зеркального отображения защищенного изображения по горизонтали. |  |

## Описание метода

В качестве основы для разрабатываемой модели выла выбрана архитектура сверточного энкодера-декодера с состязательным подходом в силу того, что модели, использующие такую архитектуру, на сегодняшний день являются передовыми в данной области.  Несмотря на результативность таких моделей, нами были введены некоторые изменения в архитектуру самой сети, также были добавлены дополнительные компоненты. Схема разработанного модуля представлена на рисунке ниже. 

<p align="center">
<img src="https://github.com/avgurov2000/Sber_marketplace/blob/main/report/ProposedModel.png" width="800"/>
</p>

## Экспериментальное исследование

Было проведено экспериментальное исследование разработанного подхода, а также сравнение его с существующими решениями. Исследование проводилось на следующих наборах данных: «DIV2k» и наборе данных товаров маркетплейсов. 

Набор данных [DIV2k](https://data.vision.ee.ethz.ch/cvl/DIV2K/) состоит из 800 фотографий для обучения и 100 для валидации. Фотографии имеют высокое разрешение, но для упрощения вычислений и сравнения моделей, разрешение было понижено до 128 на 128 пикселей. 
Набор данных [товаров маркетплейсов](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) представляет собой набор фотографий товаров маркетплейсов, размещенных на белом фоне. Всего в наборе 44000 фотографий, для тренировки было случайно отобрано 80% набора, для тестирования 20%.
В процессе обучения и тестирования в цифровые изображения поочередно встраивалась метка длиной 30, 64 и 90 бит, представляющая собой бинарное сообщение. Защищенное изображение изменялось в соответствии с перечисленными выше аугментациями, затем с такого сообщения извлекалось зашифрованное сообщение. Метрикой являлись значения точности извлеченного сообщения, а для изображения, зашифрованного меткой, вычислялось значение PSNR между ним и исходным изображением.
Метрика точности (accuracy) для извлеченного сообщения из изображений разных наборов данных, указаны в сравнительной таблице 16.3. Рассматриваемые модели были обучены с применением указанных выше аугментациям, а потом протестированы относительно каждой аугментации отдельно (столбцы Identity, Crop, Cropout, Dropout, Jpeg, Rotate, Hflip) и ко всем аугментациям (столбец CN). Наилучшая метрика для каждого значения длины сообщение выделена жирным.
