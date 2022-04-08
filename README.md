# Panoptic Segmentation(Based on Detectron2 PanopticFCN) ROS
包含3个`ros`包，由于`python2`不支持学习的环境，这里都是使用`python3`进行处理。
根据`howtorun.md`进行运行，同时查看`requirements`的环境库中，是否满足了`detectron2`的要求。一般满足`detectron2`的基本都能运行。
另外是`conda`安装`ros`的一些包，如`rospy`等。

一共包含3个`ROS wrapper`
- `pano_msg`：`ros`自定义消息包，主要包含panoptic中，图像的长，宽，语义信息，实例信息，以及类别等。
- `panoptic_ros`：主要包含分割全景分割输入输出，会输出topic用于发布分割得到的信息
- `segment_ros`：使用全景分割的输出和深度图分割的结果，实现分割的效果。

运行：需要下载pcl_catkin

# TODO
ROS包
- [x] 写一个ros包，用于全景分割及深度图的分割，实现更好的分割效果
- 全景分割实现分割后，会将分割信息通过topic发出，

**pano_visual**的效果图：
![](https://raw.githubusercontent.com/zhuhu00/img/master/Peek%202022-02-24%2021-42.gif)
使用[PanopticFCN](https://github.com/dvlab-research/PanopticFCN)进行构建

一帧图像推理时长大概为0.06s左右。
- 将原始的参数写入了config.py文件中，这里只需要设置好模型文件地址，就可以使用了。

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

