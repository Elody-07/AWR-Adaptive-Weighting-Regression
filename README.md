## AWR: Adaptive Weighting Regression for 3D Hand Pose Estimation

This is the official repository for AAAI 2020 paper [AWR: Adaptive Weighting Regression for 3D Hand Pose Estimation](https://www.aaai.org//Papers//AAAI//2020GB//AAAI-HuangW.4059.pdf). by Weiting Huang\*, Pengfei Ren\*, Jingyu Wang, Qi Qi, Haifeng Sun  (* denotes equal contribution)

Codes are implemented with Python 3.7.0 and Pytorch 1.4.0.

### Introduction

We propose an adaptive weighting regression (AWR) method to leverage the advantages of both detection-based and regression-based method. Hand joint coordinates are estimated as discrete integration of all pixels in dense representation, guided by adaptive weight maps. This learnable aggregation  process  introduces  both  dense  and  joint  supervision that allows end-to-end training and brings adaptability to weight maps, making network more accurate and robust. 



![](https://cdn.jsdelivr.net/gh/Elody-07/PicBed/20200428164654.png)

<div align=center> Fig 1. Main idea of AWR.</div>


![](https://cdn.jsdelivr.net/gh/Elody-07/PicBed/20200428164723.png)

<div align=center> Fig 2. Framework of AWR</div>


### Code Setup

We provide result on [NYU dataset](https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm) with Resnet18 (`resnet_18_uvd.txt`) and inferencing code. 

1. Download  [NYU dataset](https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm) and put `train` and `test` directory in `./data/nyu`. We also provide hand center trained using a separate 2DCNN. 
2. `pip install -r requirements.txt`
4. Modify `./config.py` according to your setting.
5. Run code `python test.py`

New: We provide HourGlass network and a pretrained model on HourGlass-1stage. You can set `load_model='./results/hourglass_1.pth'` to run the code.

### Citation

If you find our work useful in your research, please citing:

```
@inproceedings{awr,
  title={AWR: Adaptive Weighting Regression for 3D Hand Pose Estimation},
  author={Weiting Huang and Pengfei Ren and Jingyu Wang and Qi Qi and Haifeng Sun},
  booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
  year={2020}
}
```

