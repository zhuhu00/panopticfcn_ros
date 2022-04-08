#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Hu Zhu, zhuh2020@mail.sustech.edu.cn, SUSTech
import multiprocessing as mp
import numpy as np
import cv2
import torch
import threading
import sys
print(sys.version)
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog

import rospy
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
from pano_msg.msg import Panoptic
from message_filters import ApproximateTimeSynchronizer, Subscriber



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
        self.skip_frame = rospy.get_param('~skip_frames',0)
        self.counter = self.skip_frame

        rospy.loginfo('skip %d frame' % self.counter)
        # 得到root的路径，和权重路径
        # root_path = rospy.get_param('~root_path',ROOT_PATH)
        self._model_path = rospy.get_param('~MODEL_PATH',"/home/xxx.pth")
        # config_path = rospy.get_param('~config_path', CONFIG_PATH)

        # 得到输入的topic
        self.rgb_sub_ = Subscriber('/camera/rgb/image_raw', Image)
        self.depth_sub_ = Subscriber('/camera/depth/image_raw', Image)
        self.camera_info_sub_ = Subscriber('/camera/rgb/camera_info', CameraInfo)

        self.syn_sub_ = ApproximateTimeSynchronizer([self.rgb_sub_, self.depth_sub_, self.camera_info_sub_], queue_size=10, slop=0.2)

        self.syn_sub_.registerCallback(self.perceive_)

        # pub topic 
        self.pano_info_pub = rospy.Publisher("/panoptic/seg", Panoptic, queue_size=1)
        self.rgb_pub = rospy.Publisher("/panoptic/rgb_image", Image, queue_size=1)
        self.depth_pub = rospy.Publisher("/panoptic/depth_image", Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher("/panoptic/camera_info", CameraInfo, queue_size=1)
        self.pano_visual_pub = rospy.Publisher("/panoptic/pano_visual",Image, queue_size=1)

        # visual
        self._panoVisualization = rospy.get_param('~visualize_panoptic', True)
        self._rawVisualization = rospy.get_param('~visualize_raw', True)

        self._publish_rate = rospy.get_param('~publish_rate', 100)
        self._msg_lock = threading.Lock()
        self._last_msg = None
        self._cfg = setup_cfg_new(self._model_path)
        self.metadata = MetadataCatalog.get(
            self._cfg.DATASETS.TEST[0] if len(self._cfg.DATASETS.TEST) else "__unused"
        )
        self._model = DefaultPredictor(self._cfg)
        # DefaultPredictor

        self.rgb_image_ = None
        self.depth_image_ = None
        self.camera_info_ = None
        self.flag_ = False  # if new message comes


    def perceive_(self, rgb_img, depth_img, camera_info):
        # rospy.logwarn("Get an image")
        # self.counter -= 1
        # if self.counter >= 0:
        #     return
        # self.counter = self.skip_frame 
        
        self.rgb_image_ = rgb_img
        self.depth_image_ = depth_img
        self.camera_info_ = camera_info

        if self.flip_depth_:
            # dep_img = self.bridge_.imgmsg_to_cv2(depth_img, desired_encoding="16UC1")
            dep_img = self._bridge.imgmsg_to_cv2(depth_img, desired_encoding="16UC1")
            dep_flip_img = cv2.flip(dep_img, 0)
            depth_flip_img_ = self._bridge.cv2_to_imgmsg(dep_flip_img, encoding="16UC1")
            depth_flip_img_.header = depth_img.header

        # Publish frame
        self.rgb_pub.publish(rgb_img)
        self.camera_info_pub.publish(camera_info)

        if self.flip_depth_:
            self.depth_pub.publish(depth_flip_img_)
        else:
            self.depth_pub.publish(depth_img)

        self.flag_ = True


    def imgmsg_to_cv2(self, img_msg, dtype=np.dtype("uint8")):
        # print(img_msg.encoding)
        if img_msg.encoding != "rgb8":
            rospy.logwarn_once("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
        # if(dtype==np.dtype("uint8")):
        #     dtype==np.dtype("uint8")
        # else

        # dtype = np.dtype("uint8") # Hardcode to 8 bits...
        # dtype = np.dtype("uint8") # Hardcode to 8 bits...
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
        mp.set_start_method("spawn", force=True)
        while not rospy.is_shutdown():

            if self.rgb_image_ is not None and self.flag_:
                self.flag_=False
                # start = rospy.get_time()
                # img = self._bridge.imgmsg_to_cv2(self.rgb_image_)
                img = self.imgmsg_to_cv2(self.rgb_image_)
                height, width, channels = img.shape
                time = self.rgb_image_.header.stamp
                start = rospy.get_time()
                # resp = VisualizationDemo.run_predictor(img)
                resp = self._model(img)
                end = rospy.get_time()
                print("===Panoptic=Time===" ,end-start)

                self.pub_pano_info_(resp, time)
                self.pub_pano_visual(resp, time,img)

    def pub_pano_visual(self, predictions, time_, image):
        cpu_device = torch.device("cpu")
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=ColorMode.SEGMENTATION)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(panoptic_seg.to(cpu_device), segments_info)
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)
        # if not vis_output or vis_output == [None]:
        #     self.pano_visual_pub.publish(image_msg)
        # else:
        cv_result = vis_output.get_image()[:,:,::-1]
        image_msg = self.cv2_to_imgmsg(cv_result)
        self.pano_visual_pub.publish(image_msg)


    def pub_pano_info_(self, resp, time_):
        rate = rospy.Rate(self._publish_rate)
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
                        
        self.pano_info_pub.publish(pano_msg)
        rate.sleep()

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
    node.launch()
    print("====end===")

if __name__ == '__main__':
    main()