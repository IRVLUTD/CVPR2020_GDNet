#!/usr/bin/env python
"""ROS image listener"""

import os, sys
import glob
import threading
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime
# from scipy.io import savemat

import rospy
import tf
import tf2_ros
import message_filters
from tf.transformations import quaternion_matrix
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud
from geometry_msgs.msg import Pose, PoseArray, Point
from cv_bridge import CvBridge, CvBridgeError
# from ros_utils import ros_qt_to_rt, ros_pose_to_rt

# from utils_segmentation import visualize_segmentation
# from grasp_utils import compute_xyz
import ros_numpy
lock = threading.Lock()


class ImageListener:

    def __init__(self, camera='Fetch'):

        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None

        # initialize a node
        # self.tf_listener = tf.TransformListener()        

        if camera == 'Fetch':
            self.base_frame = 'base_link'
            rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
            self.camera_frame = 'head_camera_rgb_optical_frame'
            self.target_frame = self.base_frame
        elif camera == 'Gazebo':
            print(f"gazebo case")
            self.base_frame = 'base_link'
            rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
            print(f"im gonna wait fo camera ifno")
            msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
            print(f"waiting done")
            self.camera_frame = 'head_camera_rgb_optical_frame'
            self.target_frame = self.base_frame         
        elif camera == 'Realsense':
            # use RealSense camera
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
            self.camera_frame = 'measured/camera_color_optical_frame'
            self.target_frame = self.base_frame
        elif camera == 'Azure':
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/k4a/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/k4a/depth_to_rgb/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/k4a/rgb/camera_info', CameraInfo)
            self.camera_frame = 'rgb_camera_link'
            self.target_frame = self.base_frame
        else:
            print('camera %s is not supported in image listener' % camera)
            sys.exit(1)

        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.intrinsics = intrinsics
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        print(intrinsics)

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)


    def callback_rgbd(self, rgb, depth):
    
        # get camera pose in base
        # try:
        #      trans, rot = self.tf_listener.lookupTransform(self.base_frame, self.camera_frame, rospy.Time(0))
        #     #  RT_camera = ros_qt_to_rt(rot, trans)
        # except (tf2_ros.LookupException,
        #         tf2_ros.ConnectivityException,
        #         tf2_ros.ExtrapolationException) as e:
        #     rospy.logwarn("Update failed... " + str(e))
        #     RT_camera = None 
        RT_camera = None             

        if depth.encoding == '32FC1':
            depth_cv = ros_numpy.numpify(depth)
        elif depth.encoding == '16UC1':
            depth_cv = ros_numpy.numpify(depth)
            depth_cv = depth_cv.copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = ros_numpy.numpify(rgb)
        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.height = depth_cv.shape[0]
            self.width = depth_cv.shape[1]
            self.RT_camera = RT_camera


    # def get_data(self):

    #     with lock:
    #         if self.im is None:
    #             return None, None, None, None, None, self.intrinsics
    #         im_color = self.im.copy()
    #         depth_image = self.depth.copy()
    #         rgb_frame_id = self.rgb_frame_id
    #         rgb_frame_stamp = self.rgb_frame_stamp
    #         RT_camera = self.RT_camera.copy()
    #     print(f"hhhh")
    #     xyz_image = compute_xyz(depth_image, self.fx, self.fy, self.px, self.py, self.height, self.width)
    #     xyz_array = xyz_image.reshape((-1, 3))
    #     xyz_base = np.matmul(RT_camera[:3, :3], xyz_array.T) + RT_camera[:3, 3].reshape(3, 1)
    #     xyz_base = xyz_base.T.reshape((self.height, self.width, 3))
    #     return im_color, depth_image, xyz_image, xyz_base, RT_camera, self.intrinsics
    
    def get_data(self):

        with lock:
            if self.im is None:
                return None, None, None, None, None, self.intrinsics
            im_color = self.im.copy()
            depth_image = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp
            RT_camera = self.RT_camera
        print(f"return data")
        # xyz_image = compute_xyz(depth_image, self.fx, self.fy, self.px, self.py, self.height, self.width)
        # xyz_array = xyz_image.reshape((-1, 3))
        # xyz_base = np.matmul(RT_camera[:3, :3], xyz_array.T) + RT_camera[:3, 3].reshape(3, 1)
        # xyz_base = xyz_base.T.reshape((self.height, self.width, 3))
        return im_color, depth_image, RT_camera, self.intrinsics



if __name__ == '__main__':
    # test_basic_img()
    rospy.init_node("test_img_listener")
    listen = ImageListener("Gazebo")
    import time
    time.sleep(3)
    print(f"start")
    count = 0
    while count < 2:
        print(f"count {count}")
        print(listen.get_data())
        time.sleep(2)
        count+=1

