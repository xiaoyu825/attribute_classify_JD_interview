# Pedestrian ttribute recognition useing pa100k for JD's interview

## Preparation
**Prerequisite: Python 3.6.8 and Pytorch 1.1.0**

1. Install [Pytorch](https://pytorch.org/)

2. Download and prepare the dataset as follow:

    ```

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

   ```
   python demo.py
   ```

</font>
#### author xiaoyu
