## Towards Principled Design of Deep Convolutional Networks: Introducing SimpNet

![SimpNet](/SimpNetV2/images/Arch2_01.jpg)

This repository contains the architectures, pretrained models, logs, etc pertaining to the SimpNet Paper (Towards Principled Design of Deep Convolutional Networks: Introducing SimpNet) : https://arxiv.org/abs/1802.06205 

#### Abstract : 

> Major winning Convolutional Neural Networks (CNNs), such as VGGNet, ResNet, DenseNet, etc, include tens to hundreds of  millions of parameters, which impose considerable computation and memory overheads. This limits their practical usage in  training and optimizing for real-world applications. On the contrary, light-weight architectures, such as SqueezeNet, are being  proposed to address this issue. However, they mainly suffer from low accuracy, as they have compromised between the processing  power and efficiency. These inefficiencies mostly stem from following an ad-hoc designing procedure. In this work, we discuss  and propose several crucial design principles for an efficient architecture design and elaborate intuitions concerning  different aspects of the design procedure. Furthermore, we introduce a new layer called *SAF-pooling* to improve the  generalization power of the network while keeping it simple by choosing best features. Based on such principles, we propose a  simple architecture called *SimpNet*. We empirically show that *SimpNet* provides a good trade-off between the  computation/memory efficiency and the accuracy solely based on these primitive but crucial principles. SimpNet outperforms the  deeper and more complex architectures such as VGGNet, ResNet, WideResidualNet \etc, on several well-known benchmarks, while  having 2 to 25 times fewer number of parameters and operations. We obtain state-of-the-art results (in terms of a balance  between the accuracy and the number of involved parameters) on standard datasets, such as CIFAR10, CIFAR100, MNIST and SVHN.

Simply put, *SimpNet* architecture is the successor to the the successful SimpleNet architecture. It is based on a series of design principles which allowed the architecture to become superior to its precessor ([*SimpleNet*](https://github.com/Coderx7/SimpleNet)) while still retaining the same number of parameters and simplicity in design and outperforming deeper and more complex architectures (2 to 25X), such as Wide Residual Networks, ResNet, FMax, etc on a series of highly compatative benchmark datasets such as CIFAR10/100, SVHN and MNIST). 


## Citation






Note: The files are being uploaded...

## Results Overview :

#### Top CIFAR10/100 results:

| **Method**              | **\#Params** |  **CIFAR10**  | **CIFAR100** |
| :---------------------- | :----------: | :-----------: | :----------: |
| VGGNet(16L) /Enhanced   |     138m     | 91.4 / 92.45  |      \-      |
| ResNet-110L / 1202L  \* |  1.7/10.2m   | 93.57 / 92.07 | 74.84/72.18  |
| SD-110L / 1202L         |  1.7/10.2m   | 94.77 / 95.09 |  75.42 / -   |
| WRN-(16/8)/(28/10)      |    11/36m    | 95.19 / 95.83 |  77.11/79.5  |
| DenseNet                |    27.2m     |     96.26     |    80.75     |
| Highway Network         |     N/A      |     92.40     |    67.76     |
| FitNet                  |      1M      |     91.61     |    64.96     |
| FMP\* (1 tests)         |     12M      |     95.50     |    73.61     |
| Max-out(k=2)            |      6M      |     90.62     |    65.46     |
| Network in Network      |      1M      |     91.19     |    64.32     |
| DSN                     |      1M      |     92.03     |    65.43     |
| Max-out NIN             |      \-      |     93.25     |    71.14     |
| LSUV                    |     N/A      |     94.16     |     N/A      |
| **SimpNet**                 |    **5.4M**     |  **95.49/95.56**  |    **78.08**     |
| **SimpNet**                 |    **8.9M**    |     **95.89**     |    **79.17**     |


#### Comparisons of performance on SVHN:

| **Method**                   | **Error rate** |
| :--------------------------- | :------------: |
| Network in Network           |      2.35      |
| Deeply Supervised Net        |      1.92      |
| ResNet (reported by  (2016)) |      2.01      |
| ResNet with Stochastic Depth |      1.75      |
| DenseNet                     |   1.79-1.59    |
| Wide ResNet                  |   2.08-1.64    |
| **SimpNet**                  |    **1.648**  |

* The slim version achieves 1.95% error rate.

#### MNIST results:

| **Method**                   | **Error rate** |
| :--------------------------- | :------------: |
| Batch-normalized Max-out NIN |     0.24%      |
| Max-out network (k=2)        |     0.45%      |
| Network In Network           |     0.45%      |
| Deeply Supervised Network    |     0.39%      |
| RCNN-96                      |     0.31%      |
| **SimpNet**                      |     **0.25%**      |

* The slim version achives 99.73% accuracy.


#### Slim Version Results on CIFAR10/100 : 

| Model                                        |    Param    |    CIFAR10    |   CIFAR100    |
| :------------------------------------------- | :---------: | :-----------: | :-----------: |
| **SimpNet**                                  |**300K - 600K**| **93.25 - 94.03** | **68.47 - 71.74** |
| Maxout                                       |     6M      |     90.62     |     65.46     |
| DSN                                          |     1M      |     92.03     |     65.43     |
| ALLCNN                                       |    1.3M     |     92.75     |     66.29     |
| dasNet                                       |     6M      |     90.78     |     66.22     |
| ResNet  <span>(Depth32, tested by us)</span> |    475K     |     93.22     |  67.37-68.95  |
| WRN                                          |    600K     |     93.15     |     69.11     |
| NIN                                          |     1M      |     91.19     |       —       |

## Principle Experiments : 

#### Gradual Expansion with Minimum Allocation:

#### Homogeneous Groups of Layers:

#### Local Correlation Preservation:

#### Maximum Information Utilization:

#### Maximum Performance Utilization:

#### Balanced Distribution Scheme:

#### Rapid Prototyping In Isolation:

#### Simple Adaptive Feature Composition Pooling:

#### Dropout Utilization:

#### Final Regulation Stage:


## Simple Adaptive Feature Composition Pooling (SAFC Pooling) :

![SAFC Pooling](https://github.com/Coderx7/SimpNet/blob/master/SimpNetV2/images/pooling_concept2%20_v5_larged_2.jpg?)

## Generalization Examples: 


