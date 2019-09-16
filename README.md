# Pedestrian attribute recognition using pa100k for JD's interview

## Preparation
**Prerequisite: Python 3.6.8 and Pytorch 1.1.0**

1. Install [Pytorch](https://pytorch.org/)

2. Download and prepare the dataset as follow:

    PA100K [Links](https://drive.google.com/drive/folders/0B5_Ra3JsEOyOUlhKM0VPZ1ZWR2M)
    ```
    ./dataset/pa100k/release_data/release_data/*.jpg      
    ./dataset/pa100k/annotation.mat
    ``` 
    ```
    python transform_pa100k.py 
    ```
## Train the model
<font face="Times New Roman" size=4>

   ```
   sh train.sh
   ``` 

## Demo 
<font face="Times New Roman" size=4>
    for saving time ,you can use my trained parameter.
  链接: https://pan.baidu.com/s/1_kP9iN2GzRRU6UOCf-y0zw 提取码: 9kgz 
    
 ```    
  ./model_parameter/ckpt_epoch3.pth
 ``` 

   ```
   python demo.py
   ```

</font>



#### @author xiaoyu
