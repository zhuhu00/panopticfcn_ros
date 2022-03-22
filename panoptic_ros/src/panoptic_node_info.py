#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Hu Zhu, zhuh2020@mail.sustech.edu.cn, SUSTech
import argparse
from asyncore import read
from distutils.command.config import config
import glob
import imp
# import imp
import multiprocessing as mp
from pickle import FALSE
from tkinter.tix import MAIN
import numpy as np
import time
import cv2
from rospkg import RosPack
import tqdm
import threading
import sys
print(sys.version)
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
# from detectron2.utils.logger import setup_logger
# from panoptic_ros.src.panoptic_node import setup_cfg_new
from detectron2.engine.defaults import DefaultPredictor

from predictor import VisualizationDemo
# import tf
import rospy
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
# from cv_bridge import CvBridge, CvBridgeError
from pano_msg.msg import Panoptic
from message_filters import ApproximateTimeSynchronizer, Subscriber


# constants
WINDOW_NAME = "COCO detections"
# Local path to trained weights file
MODEL_PATH = '/home/roma/dev/catkin_panoptic/src/panoptic_ros/weights/panoptic_fcn_r50_512_3x.pth'
# MODEL_PATH = 'xxx'
CONFIG_PATH = 'config/panoptic-demo.yaml'
RGB_TOPIC = '/camera/color/image_raw'
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: CLASS_NAMES.index('teddy bear')
CLASS_NAMES = ['BG', 'person', 'agv']
IMG_TEST = './test16.png'

def setup_cfg_new(MODEL_PATH):
    cfg = get_cfg()
    from detectron2.projects.panopticfcn.config import add_panopticfcn_config  # noqa
    add_panopticfcn_config(cfg)
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.freeze()
    return cfg


class PanopticV1Node(object):
    def __init__(self):
        # configs
        self.flip_depth_ = rospy.get_param('~filp_depth',False)
        self.skip_frame = rospy.get_param('~skip_frame',0)
        self.counter = self.skip_frame
        rospy.loginfo('skip %d frame' % self.counter)
        # 得到root的路径，和权重路径
        # root_path = rospy.get_param('~root_path',ROOT_PATH)
        self._model_path = rospy.get_param('~MODEL_PATH',MODEL_PATH)
        config_path = rospy.get_param('~config_path', CONFIG_PATH)

        # 得到输入的topic
        self.rgb_sub_ = Subscriber('/camera/rgb/image_raw', Image)
        self.depth_sub_ = Subscriber('/camera/depth/image_raw', Image)
        self.camera_info_sub_ = Subscriber('/camera/rgb/camera_info', CameraInfo)

        self.syn_sub_ = ApproximateTimeSynchronizer([self.rgb_sub_, self.depth_sub_, self.camera_info_sub_], queue_size=10, slop=0.2)

        self.syn_sub_.registerCallback(self.perceive_)

        # pub topic 
        self.dt_pub = rospy.Publisher("/perception/seg", Panoptic, queue_size=1)
        self.rgb_pub = rospy.Publisher("/perception/rgb_image", Image, queue_size=1)
        self.depth_pub = rospy.Publisher("/perception/depth_image", Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher("/perception/camera_info", CameraInfo, queue_size=1)

        # visual
        self._panoVisualization = rospy.get_param('~visualize_panoptic', True)
        self._rawVisualization = rospy.get_param('~visualize_raw', True)

        self._publish_rate = rospy.get_param('~publish_rate', 100)
        self._msg_lock = threading.Lock()
        self._last_msg = None
        self._cfg = setup_cfg_new(self._model_path)
        # self._bridge = CvBridge()
        # self._model = VisualizationDemo(self._cfg)
        self._model = DefaultPredictor(self._cfg)
        # DefaultPredictor

        self.rgb_image_ = None
        self.depth_image_ = None
        self.camera_info_ = None
        self.depth_info_ = None
        self.flag_ = False  # if new message comes


    def perceive_(self, rgb_img, depth_img, camera_info):
        # print("percieve")
            
        self.rgb_image_ = rgb_img
        self.depth_image_ = depth_img
        self.camera_info_ = camera_info

        if self.flip_depth_:
            # dep_img = self.bridge_.imgmsg_to_cv2(depth_img, desired_encoding="16UC1")
            dep_img = self._bridge.imgmsg_to_cv2(depth_img, desired_encoding="16UC1")
            dep_flip_img = cv2.flip(dep_img, 0)
            depth_flip_img_ = self._bridge.cv2_to_imgmsg(dep_flip_img, encoding="16UC1")
            depth_flip_img_.header = depth_img.header

        # img = self.bridge_.imgmsg_to_cv2(rgb_img)
        # height, width, channels = img.shape
        # if (channels == 4):
        #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # may include some other processing

        # Publish frame
        self.rgb_pub.publish(rgb_img)
        self.camera_info_pub.publish(camera_info)

        if self.flip_depth_:
            self.depth_pub.publish(depth_flip_img_)
        else:
            self.depth_pub.publish(depth_img)

        self.flag_ = True


    def imgmsg_to_cv2(self, img_msg):
        # print(img_msg.encoding)
        if img_msg.encoding != "rgb8":
            rospy.logwarn_once("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
        dtype = np.dtype("uint8") # Hardcode to 8 bits...
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                        dtype=dtype, buffer=img_msg.data)
        # If the byt order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            image_opencv = image_opencv.byteswap().newbyteorder()
        image_reverse = np.flip(image_opencv, axis=2)

        return image_reverse

    def cv2_to_imgmsg(self, cv_image):
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tobytes()
        img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
        return img_msg

    def launch(self):
        while not rospy.is_shutdown():

            if self.rgb_image_ is not None and self.flag_:
                self.flag_=False
                # start = rospy.get_time()
                # img = self._bridge.imgmsg_to_cv2(self.rgb_image_)
                img = self.imgmsg_to_cv2(self.rgb_image_)
                height, width, channels = img.shape
                time = self.rgb_image_.header.stamp
                self.pub_dt_(img, time)

    def pub_dt_(self, img, time_):
        start = rospy.get_time()
        # resp = VisualizationDemo.run_predictor(img)
        resp = self._model(img)
        end = rospy.get_time()
        print("===Panoptic=Time===" ,end-start)
        pano_msg = Panoptic()
        pano_msg.header.stamp = time_

        # panoptic ==instance
        pano_resp = resp["panoptic_seg"]
        inst_resp = resp["instances"]

        # seg_map = resp["seg_map"]
        seg_map = pano_resp[0].cpu().detach().numpy().astype("uint8")
        # h, w = seg_map.shape

        if(len(resp["instances"].pred_boxes)>=1):
        # pack semantic information
            info  = pano_resp[1]
            boxes = inst_resp.pred_boxes.tensor.cpu().detach().numpy().tolist()
        else:
            info  = pano_resp[1]
            boxes =[]

        # info  = pano_resp[1]
        # boxes = resp["boxes"]
        boxes = np.reshape(boxes, len(boxes)*4)
        obj_id = []
        sem_id = []
        scores = []
        area = []
        obj_category = []
        sem_category = []

        for i in range(len(info)):
            info_current = info[i]
            if info_current["isthing"]:
                scores.append(info_current["score"])
                obj_category.append(info_current["category_id"])
                obj_id.append(info_current["id"])  
            else:
                area.append(info_current["area"])
                sem_category.append(info_current["category_id"]+80) # start from 81, no class 80
                sem_id.append(info_current["id"])

        pano_msg.height = len(seg_map)
        pano_msg.width = len(seg_map[0])
        seg_map = seg_map.reshape(pano_msg.height*pano_msg.width)
        pano_msg.seg_map = seg_map
        pano_msg.obj_id = obj_id
        pano_msg.obj_category = obj_category
        pano_msg.obj_scores = scores
        # 2D bounding boxes [x1 y1 x2 y2 ...]
        pano_msg.obj_boxes = [int(i) for i in boxes]

        pano_msg.sem_id = sem_id
        pano_msg.sem_category = sem_category
        pano_msg.sem_area = area
                        
        self.dt_pub.publish(pano_msg)

    def run(self):
        mp.set_start_method("spawn", force=True)
        print("run ")
        self.__mask_pub = rospy.Publisher('~mask', Image, queue_size=1)
        vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)

        rospy.Subscriber(self._rgb_input_topic,Image,self._image_callback, queue_size=1)

        rate = rospy.Rate(self._publish_rate)
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                np_image = self.imgmsg_to_cv2(msg)
                print(np_image.shape)
                start_time = time.time()
                predictions, visualized_output = self._model.run_on_image(np_image)
                panoptic_time = time.time() - start_time
                rospy.logwarn_once('Time needed for segmentation: %.3f s' % panoptic_time)
                print("{}: {} in {:.2f}s".format(
            "IMG_TEST",
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            panoptic_time,))
                if self._visualization:  
                    if not visualized_output or visualized_output == [None]:
                        vis_pub.publish(msg)
                    else:
                        cv_result = visualized_output.get_image()[:,:,::-1]
                        image_msg = self.cv2_to_imgmsg(cv_result)
                        vis_pub.publish(image_msg)
            rate.sleep()

    def _build_result_msg(self, msg, result):
        result_msg = Image()
        result_msg.header = msg.header
        #print('Time needed for segmentation: %.3f s' % msg.header)
        result_msg.encoding = "mono8"
        if not result or result == [None]:
            result_msg.height = 720
            result_msg.width = 1280
            result_msg.step = result_msg.width
            result_msg.is_bigendian = False
            mask_sum = np.zeros(shape=(1280,720),dtype=np.uint8)
            result_msg.data = mask_sum.tobytes()
            return result_msg
        cur_result = result[0]
        seg_label = cur_result[0]
        seg_label = seg_label.cpu().numpy().astype(np.uint8)
        cate_label = cur_result[1]
        cate_label = cate_label.cpu().numpy()
        score = cur_result[2].cpu().numpy()


        vis_inds = score > self._score_thr
        seg_label = seg_label[vis_inds]
        result_msg.height = seg_label.shape[1]
        result_msg.width = seg_label.shape[2]
        result_msg.step = result_msg.width
        result_msg.is_bigendian = False
        num_mask = seg_label.shape[0]
        cate_label = cate_label[vis_inds]
        cate_score = score[vis_inds]

        mask_sum = np.zeros(shape=(result_msg.height,result_msg.width),dtype=np.uint8)
        for i in range(num_mask):
            class_id = cate_label[i] # 0,1,2
            class_name = self._class_names[class_id]  #class name
            score = cate_score[i] #correponding score
            mask_sum += seg_label[i, :, :] * (class_id+1)
            # if class_id==1:
            #     mask_sum += seg_label[i, :, :] * (class_id+1)*20 + seg_label[i, :, :] * 150
            # else:
            #     mask_sum += seg_label[i, :, :] * (class_id+1)*20 + seg_label[i, :, :] * 50

        result_msg.data = mask_sum.tobytes()
        return result_msg

    def _image_callback(self, msg):
        rospy.logwarn("Get an image")
        self.counter -= 1
        if self.counter >= 0:
            return
        self.counter = self.skip_frame 
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()

def main():
    rospy.init_node('panoptic_v1')
    

    node = PanopticV1Node()
    # node.run()
    node.launch()
    print("====end===")

def test():
# main()
    # 设置进程的启动方式
    mp.set_start_method("spawn", force=True)
    # 解析参数
    # args = get_parser().parse_args()

    # setup_logger(name="fvcore")
    # # 开始记录日志，并打印相关的日志
    # logger = setup_logger()
    # logger.info("Arguments: " + str(args))

    # 将args设置到config上
    # cfg = setup_cfg(args)
    cfg = setup_cfg_new()
    # logger.info("cfgs: === " + str(cfg)+"===")

    # 显示
    img = read_image(IMG_TEST, format="BGR")
    demo = VisualizationDemo(cfg)
    # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # cv2.imshow(WINDOW_NAME, img)
    # cv2.waitKey(0)

    start_time = time.time()
    # predictions, visualized_output = demo.run_on_image(img)
    predictions, vis_panoptic_output = demo.run_on_image(img)
    print(
        "{}: {} in {:.2f}s".format(
            IMG_TEST,
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )

    # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow("vis_panoptic_output", vis_panoptic_output.get_image()[:,:,::-1])
    cv2.waitKey(0)
    # vis_panoptic_output.save("./16out.png")

if __name__ == '__main__':
    main()
    # test()