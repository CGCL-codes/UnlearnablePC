# UnlearnablePC
The official implementation of our NeurIPS 2024 paper "*[Unlearnable 3D Point Clouds: Class-wise Transformation Is All You Need]()*", by *[Xianlong Wang](https://wxldragon.github.io/), [Minghui Li](http://trustai.cse.hust.edu.cn/index.htm), [Wei Liu](https://wilmido.github.io/), [Hangtao Zhang](https://scholar.google.com.hk/citations?user=H6wMyNEAAAAJ&hl=zh-CN), [Shengshan Hu](http://trustai.cse.hust.edu.cn/index.htm), [Yechao Zhang](https://scholar.google.com.hk/citations?user=6DN1wxkAAAAJ&hl=zh-CN&oi=ao), [Ziqi Zhou](https://zhou-zi7.github.io/), and [Hai Jin](https://scholar.google.com.hk/citations?user=o02W0aEAAAAJ&hl=zh-CN&oi=ao).*

![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-blue.svg?style=plastic) 
![Unlearnable Examples](https://img.shields.io/badge/Unlearnable-Examples-yellow.svg?style=plastic)
![3D Point Clouds](https://img.shields.io/badge/3D-Point-Clouds-orange.svg?style=plastic)
 


## Abstract
Traditional unlearnable strategies have been proposed to prevent unauthorized users from training on the 2D image data. With more 3D point cloud data containing sensitivity information, unauthorized usage of this new type data has also become a serious concern. To address this, we propose the first integral unlearnable framework for 3D point clouds including two processes: (i) we propose an unlearnable data protection scheme, involving a class-wise setting established by a category-adaptive allocation strategy and multi-transformations assigned to samples; (ii) we propose a data restoration scheme that utilizes class-wise inverse matrix transformation, thus 
enabling authorized-only training for unlearnable data. This restoration process is a practical issue overlooked in most existing unlearnable literature, i.e., even authorized users struggle to gain knowledge from 3D unlearnable data. Both theoretical and empirical results (including 6 datasets, 16 models, and 2 tasks) demonstrate the effectiveness of our proposed unlearnable framework. 
<p align="center">
  <img src="unlearnablepc.png" width="700"/>
</p>


## Latest Update
| Date       | Event    |
|------------|----------|
| **2024/09/26** | ECLIPSE is acccepted by ESORICS 2024!  |
