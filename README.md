<h2 align="center">Защита визуального контента маркетплейсов путем стеганографии в цифровых изображениях</h2>
<h5 align="center">НЦКР ITMO, Санкт-Петербург, Россия</h5>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


## Содержание



## Выбор аугментации

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

## Разработанный метод

В качестве основы для разрабатываемой модели выла выбрана архитектура сверточного энкодера-декодера с состязательным подходом в силу того, что модели, использующие такую архитектуру, на сегодняшний день являются передовыми в данной области [[1](#1), [2](#2), [3](#3)].  Несмотря на результативность таких моделей, нами были введены некоторые изменения в архитектуру самой сети, также были добавлены дополнительные компоненты. Схема разработанного модуля представлена на рисунке ниже. 

<p align="center">
<img src="https://github.com/avgurov2000/Sber_marketplace/blob/main/report/ProposedModel.png" width="800"/>
<caption="Схема разработанного модуля">
</p>

Основные изменения коснулись строения энкодера. Хотя авторы, использующие данную архитектуру [[1](#1), [2](#2), [3](#3), [6](#6)], и утверждают, что использование последовательных блоков, состоящих из блока свертки, пакетной нормализации [[7]](#7) и функции активации [[8]](#8), является зарекомендовавшим себя решением, все же это не уберегает их модели от деградации точности по причине потенциальной высокой симметрии пространства параметров модели. Поэтому нами были добавлены остаточные связи [[9]](#9) на каждом счетном блоке, а также доступ к первоначальному цифровому изображению и встраиваемому сообщению на каждом нечетном. Архитектура такого энкодера из семи блоков представлена на рисунке ниже. 

<p align="center">
<img src="https://github.com/avgurov2000/Sber_marketplace/blob/main/report/encoder.png" width="720"/>
</p>

Другой отличительной чертой стало добавление дополнительной составляющей целевой функции (на схеме метода обозначена как L с индексом SSL), которая отвечает за то, чтобы извлеченное сообщение с аугментированного изображения походила на сообщение, извлеченное с оригинального защищенного изображения. Делается это путем дистилляции знаний, используемой часто в подходах самообучения [[10]](#10).

Использование дискриминатора как соревновательной составляющей также обсуждалась в работах ранее [[1]](#1) и хорошо себя зарекомендовало, однако, его устоявшаяся архитектура, широко использовавшаяся в таких моделях, была склонна к коллапсу, то есть быстрому обнулению ошибки дискриминатора при обучении. Решением этому послужило избавление от смещения в блоках дискриминатора, как это часто делается на практике [[11]](#11). 


## Экспериментальное исследование

Было проведено экспериментальное исследование разработанного подхода, а также сравнение его с существующими решениями. Исследование проводилось на следующих наборах данных: «DIV2k» и наборе данных товаров маркетплейсов. 

Набор данных [DIV2k](https://data.vision.ee.ethz.ch/cvl/DIV2K/) [[4]](#4) состоит из 800 фотографий для обучения и 100 для валидации. Фотографии имеют высокое разрешение, но для упрощения вычислений и сравнения моделей, разрешение было понижено до 128 на 128 пикселей. 

Набор данных [товаров маркетплейсов](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) [[5]](#5) представляет собой набор фотографий товаров маркетплейсов, размещенных на белом фоне. Всего в наборе 44000 фотографий, для тренировки было случайно отобрано 80% набора, для тестирования 20%.

В процессе обучения и тестирования в цифровые изображения поочередно встраивалась метка длиной 30, 64 и 90 бит, представляющая собой бинарное сообщение. Защищенное изображение изменялось в соответствии с перечисленными выше аугментациями, затем с такого сообщения извлекалось зашифрованное сообщение. Метрикой являлись значения точности извлеченного сообщения, а для изображения, зашифрованного меткой, вычислялось значение PSNR между ним и исходным изображением.

Метрика точности (accuracy) для извлеченного сообщения из изображений разных наборов данных, указаны в сравнительной таблице ниже. Рассматриваемые модели были обучены с применением указанных выше аугментациям, а потом протестированы относительно каждой аугментации отдельно (столбцы Identity, Crop, Cropout, Dropout, Jpeg, Rotate, Hflip) и ко всем аугментациям (столбец CN). Наилучшая метрика для каждого значения длины сообщение выделена жирным.

<p align="center">
<img src="https://github.com/avgurov2000/Sber_marketplace/blob/main/report/comp_table.png" width="800"/>
</p>


## Ссылки

<a id="1">[1]</a> 
Zhu J. et al. HiDDeN: Hiding Data With Deep Networks. 2018.

<a id="2">[2]</a>
Zhang H. et al. Robust Data Hiding Using Inverse Gradient Attention. 2020.

<a id="3">[3]</a>
Hamamoto I., Kawamura M. Neural Watermarking Method Including an Attack Simulator against Rotation and Compression Attacks // IEICE Trans Inf Syst. The Institute of Electronics, Information and Communication Engineers, 2020. Vol. E103.D, № 1. P. 33–41.

<a id="4">[4]</a> 
Agustsson E., Timofte R. NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study. 2017. P. 126–135.

<a id="5">[5]</a> 
Fashion Product Images Dataset | Kaggle [Electronic resource]. URL: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset (accessed: 14.12.2022).

<a id="6">[6]</a> 
Luo X. et al. Distortion Agnostic Deep Watermarking // Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition. IEEE Computer Society, 2020. P. 13545–13554.

<a id="7">[7]</a> 
Ioffe S., Szegedy C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. PMLR, 2015. P. 448–456.

<a id="8">[8]</a> 
Agarap A.F. Deep Learning using Rectified Linear Units (ReLU). 2018

<a id="9">[9]</a> 
He K. et al. Deep Residual Learning for Image Recognition. 2016. P. 770–778.

<a id="10">[10]</a> 
Caron M. et al. Emerging Properties in Self-Supervised Vision Transformers // Proceedings of the IEEE International Conference on Computer Vision. Institute of Electrical and Electronics Engineers Inc., 2021. P. 9630–9640.

<a id="11">[11]</a>
Radford A., Metz L., Chintala S. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks // 4th International Conference on Learning Representations, ICLR 2016 - Conference Track Proceedings. International Conference on Learning Representations, ICLR, 2015.
