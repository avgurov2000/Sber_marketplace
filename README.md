<h2 align="center">Защита визуального контента маркетплейсов путем стеганографии в цифровых изображениях</h2>
<h5 align="center">НЦКР ITMO, Санкт-Петербург, Россия</h5>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


## Содержание



## Аугментации

В качестве задачи стоит разработка метода встраивания цифровых водяных знаков в изображения товаров маркетплейсов, наибольший интерес вызывает устойчивость такого метода защиты к геометрическим аугментациям защищенных изображений. В связи с чем нами был определен набор аугментация/шума, относительно которого будет проводиться обучение и тестирование моделей. Набор аугментаций и их параметров перечислен ниже в таблице:

| Аугментация       | Описание                                        | Параметры        |
| -------------     |-------------                                    | -------------    |
| Identity          | В этом случае изображение с встроенной меткой никак не изменяется. Использование не измененного изображения в качестве атаки при обучение призвано  предотвратить деградацию точности при извлечения водяного знака.   | |
| Crop              | Данный вид атаки обрезает изображение с защитной меткой до части меньшего размера. | Соотношение размеров исходного и обрезанного изображения, выбирается случайно из диапазона [0.25, 1].|
| Cropout           | Данный вид атаки используется для комбинирования части изображения с защищенной меткой и оставшейся части без него. | Соотношение размера исходного изображения и размера части изображения со встроенной меткой, выбирается случайно из диапазона [0.25, 1]. |
| Dropout           | Данный вид атаки используется для комбинирования части изображения с защищенной меткой и оставшейся части без него. В отличии от атаки Cropout, часть изображения с защищенной меткой в этом случае представляет собой не определенный вырезанный прямоугольный кусок защищенного изображения, а случайно выбранные пиксели такого изображения. Соответственно, оставшиеся пиксели представляют собой пиксели незащищенного изображения. | Соотношение пикселей защищенного изображения к количеству всех пикселей изображения, выбирается случайно из диапазона [0.75, 1].| 
| Jpeg              | При данной атаки к исходному защищенному изображению применятся jpeg-компрессия. |  | 
| Rotate            | Данная атака используется для поворота защищенного изображения на определенный угол. | Угол поворота в градусах, выбирается случайно из диапазона [-15, 15]. |
| Hflip             | Данная атака используется для зеркального отображения защищенного изображения по горизонтали. |  |

## Описание метода

В качестве основы для разрабатываемой модели выла выбрана архитектура сверточного энкодера-декодера с состязательным подходом в силу того, что модели, использующие такую архитектуру, на сегодняшний день являются передовыми в данной области.  Несмотря на результативность таких моделей, нами были введены некоторые изменения в архитектуру самой сети, также были добавлены дополнительные компоненты. Схема разработанного модуля представлена на рисунке ниже. 

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Схема разработанного модуля"
