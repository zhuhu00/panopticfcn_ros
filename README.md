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


how to run origin model:
```bash
cd /path/to/detectron2
python3 projects/PanopticFCN/train.py --config-file <config.yaml> --num-gpus 8 --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```

# 原始的是如何运行的呢？
出现错误的解决办法
![](https://raw.githubusercontent.com/zhuhu00/img/master/2022-03-22-22-34-08.png)
这个是由于`yacs`版本不对导致的可以由如下解决：
```bash
pip uninstall yacs
pip install yacs --upgrade
```

# 运行
```bash
python3 demo/demo.py --config-file projects/PanopticFCN/configs/PanopticFCN-R50-3x-FAST.yaml --input /../listfolder/*.png --output /../output/
```

修改demo里面的
```
from detectron2.projects.panopticfcn.config import add_panopticfcn_config  # noqa
add_panopticfcn_config(cfg)

```

