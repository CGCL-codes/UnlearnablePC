# UnlearnablePC
The official implementation of our NeurIPS 2024 paper "*[Unlearnable 3D Point Clouds: Class-wise Transformation Is All You Need](https://arxiv.org/pdf/2410.03644)*", by *[Xianlong Wang](https://wxldragon.github.io/), [Minghui Li](http://trustai.cse.hust.edu.cn/index.htm), [Wei Liu](https://wilmido.github.io/), [Hangtao Zhang](https://scholar.google.com.hk/citations?user=H6wMyNEAAAAJ&hl=zh-CN), [Shengshan Hu](http://trustai.cse.hust.edu.cn/index.htm), [Yechao Zhang](https://scholar.google.com.hk/citations?user=6DN1wxkAAAAJ&hl=zh-CN&oi=ao), [Ziqi Zhou](https://zhou-zi7.github.io/), and [Hai Jin](https://scholar.google.com.hk/citations?user=o02W0aEAAAAJ&hl=zh-CN&oi=ao).*

![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-blue.svg?style=plastic) 
![Unlearnable Examples](https://img.shields.io/badge/Unlearnable-Examples-yellow.svg?style=plastic)
![3D Point Clouds](https://img.shields.io/badge/Point-Clouds-orange.svg?style=plastic)
 


## Abstract
Traditional unlearnable strategies have been proposed to prevent unauthorized users from training on the 2D image data. With more 3D point cloud data containing sensitivity information, unauthorized usage of this new type data has also become a serious concern. To address this, we propose the first integral unlearnable framework for 3D point clouds including two processes: (i) we propose an unlearnable data protection scheme, involving a class-wise setting established by a category-adaptive allocation strategy and multi-transformations assigned to samples; (ii) we propose a data restoration scheme that utilizes class-wise inverse matrix transformation, thus 
enabling authorized-only training for unlearnable data. This restoration process is a practical issue overlooked in most existing unlearnable literature, i.e., even authorized users struggle to gain knowledge from 3D unlearnable data. Both theoretical and empirical results (including 6 datasets, 16 models, and 2 tasks) demonstrate the effectiveness of our proposed unlearnable framework. 
<p align="center">
  <img src="unlearnablepc.png" width="700"/>
</p>


## Latest Update
| Date       | Event    |
|------------|----------|
| **2024/10/21** | We have released the official implementation of UnlearnablePC!  |
| **2024/10/07** | The paper (arXiv version) is available at [UnlearnablePC](https://arxiv.org/pdf/2410.03644)!|
| **2024/09/26** | UnlearnablePC is acccepted by NeurIPS 2024!  |

## Start Running UnlearnablePC
- **Get code**
```shell 
git clone https://github.com/CGCL-codes/UnlearnablePC.git
```

- **Build environment**
```shell
cd UnlearnablePC
conda create -n UnlearnablePC python=3.9
conda activate UnlearnablePC
pip install -r requirements.txt
```

- **Download datasets**
  - Please download ModelNet dataset at: [[ModelNet](https://www.kaggle.com/datasets/chenxaoyu/modelnet-normal-resampled)], ShapeNetPart dataset at: [[ShapeNetPart](https://www.kaggle.com/datasets/mitkir/shapenet)], ScanObjectNN dataset at: [[ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)], S3DIS dataset at: [[S3DIS](http://buildingparser.stanford.edu/dataset.html)] (for semantic segmentation task)
  - Unzip the datasets.zip files in ```UnlearnablePC/clean_data```


- **Produce unlearnable 3D point cloud datasets**
 - Using UMT (k=4) with default settings and parameters (robust unlearnable dataset)
 ```shell
 python UnlearnablePC_generation.py --dataset ModelNet10 --mode "['rot', 'scale', 'twist', 'shear']"
 ```
 - Using UMT (k=3) with default settings and parameters (robust unlearnable dataset)
 ```shell
 python UnlearnablePC_generation.py --dataset ModelNet10 --mode "['rot', 'scale', 'twist']"
 ```
 - Using UMT (k=2) with default settings and parameters (fragile to SE(3)-invariance networks)
 ```shell
 python UnlearnablePC_generation.py --dataset ModelNet10 --mode "['rot', 'scale']"
 ```

- **Training on UMT datasets (i.e., unlearnable datasets)**
```shell
python train.py --dataset ModelNet10 --target_model pointnet_cls --UMTK 3
```

- **Training on clean datasets (i.e., raw 3D point cloud datasets)**
```shell
python train.py --dataset ModelNet10 --target_model pointnet_cls --clean_train
```

- **Employing the data restoration scheme to UMT data (for authorized users)**
```shell
python reverse_train.py --dataset ModelNet10 --target_model pointnet_cls --mode "['rot', 'scale', 'twist']"
```

## BibTex
If you find UnlearnablePC both interesting and helpful, please consider citing us in your research or publications:
```bibtex
@inproceedings{wang2024unlearnable,
  title={Unlearnable 3D Point Clouds: Class-wise Transformation Is All You Need},
  author={Wang, Xianlong and Li, Minghui and Liu, Wei and Zhang, Hangtao and Hu, Shengshan and Zhang, Yechao and Zhou, Ziqi and Jin, Hai},
  booktitle={Proceedings of the 38th Annual Conference on Neural Information Processing Systems (NeurIPS'24)},
  year={2024}
} 
```





