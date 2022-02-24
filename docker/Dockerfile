# Copyright (C) 2021, Hu Zhu, SUSTech
# email: zhuhu00@foxmail.com

FROM zhuhu/cuda102-ros-ubuntu18:0.1


#  ===========以下为自己可以添加的内容
# 以上构建成了docker镜像,可在官网拉取
# 根据不同的环境,可以在这里安装对应的依赖
RUN pip install protobuf==3.17.3 colorama
RUN conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
RUN conda install -c conda-forge addict rospkg pycocotools
RUN pip install alfred-py scipy==1.7.0 pillow opencv-python


# 创建ros空间, create catkin workspace
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws
RUN bash -c "source /opt/ros/melodic/setup.bash && \
             catkin_make -DCMAKE_BUILD_TYPE=Release"

# Load ROS environment at docker exec bash
RUN echo -e "\nsource /opt/ros/melodic/setup.bash" >> /root/.bashrc
RUN echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc
RUN echo -e "\nPS1='\[\e[1;35m\]\u@\h:\[\e[0m\]\[\e[1;33m\]\w\[\e[1;35m\]\[\e[0m\]\[\e[1;34m\]\$ \[\e[0m\]'" >> /root/.bashrc
##############################################

# # Build
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash && catkin_make -DCMAKE_BUILD_TYPE=Release"


# # Load ROS environment at each run
COPY ./ros_entrypoint.sh /
RUN chmod 755 /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]

CMD ["bash"]

WORKDIR /root/catkin_ws/src
