# TODO
ROS包
- [x] 写一个ros包，用于全景分割

效果图：
![Peek 2022-02-24 21-42](https://raw.githubusercontent.com/zhuhu00/img/master/Peek%202022-02-24%2021-42.gif)
使用[PanopticFCN](https://github.com/dvlab-research/PanopticFCN)进行构建

一帧图像推理时长大概为0.15s。

# Prerequisites
- pytorch>=1.8  detectron2的安装
    
    ```bash
    cd Thirdparty/detectron2
    python setup.py install
    ```
    
# RUN 
```bash
catkin_make
source devel/setup.bash
roslaunch panoptic_ros panoptic_ros.launch
```