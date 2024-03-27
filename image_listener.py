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
from sensor_msgs.msg import Image, CameraInfo, PointCloud, LaserScan
from geometry_msgs.msg import Pose, PoseArray, Point
from cv_bridge import CvBridge, CvBridgeError
from ros_utils import ros_qt_to_rt, ros_pose_to_rt

# from utils_segmentation import visualize_segmentation
from grasp_utils import compute_xyz
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
        self.tf_listener = tf.TransformListener()      

        if camera == 'Fetch':
            self.base_frame = 'base_link'
            rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
            self.camera_frame = 'head_camera_rgb_optical_frame'
            self.target_frame = self.base_frame
            rospy.Subscriber("/base_scan", LaserScan, self.callback_laserscan)

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
        try:
            #  print("callback")
             trans, rot = self.tf_listener.lookupTransform(self.base_frame, self.camera_frame, rospy.Time(0))
             RT_camera = ros_qt_to_rt(rot, trans)
            #  print(f"cam trans {trans}, cam rotation {rot}\n")
             self.trans_l, self.rot_l = self.tf_listener.lookupTransform(self.base_frame, 'laser_link', rospy.Time(0))
            #  print(f"laser trans {trans_l}, laser rotation {rot_l}\n")
             RT_laser = ros_qt_to_rt(self.rot_l, self.trans_l)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Update failed... " + str(e))
            RT_camera = None
            RT_laser = None             

        if depth.encoding == '32FC1':
            # depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
            depth_cv = ros_numpy.numpify(depth)
        elif depth.encoding == '16UC1':
            # depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            depth_cv = ros_numpy.numpify(depth)
            depth_cv = depth_cv.copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        # im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        im = ros_numpy.numpify(rgb)
        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.height = depth_cv.shape[0]
            self.width = depth_cv.shape[1]
            self.RT_camera = RT_camera
            self.RT_laser = RT_laser

    def callback_laserscan(self, scan):
        # print("scan received")
        self.revised_scan = scan
        self.laserscan = {
        'angle_min': scan.angle_min,
        'angle_max': scan.angle_max,
        'angle_increment': scan.angle_increment,
        'time_increment': scan.time_increment,
        'scan_time': scan.scan_time,
        'range_min': scan.range_min,
        'range_max': scan.range_max,
        'ranges': np.array(scan.ranges),
        'intensities': np.array(scan.intensities),
        }

    def get_data(self):

        with lock:
            if self.im is None:
                return None, None, None, None, None, self.intrinsics
            im_color = self.im.copy()
            depth_image = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp
            RT_camera = self.RT_camera.copy()
            RT_laser = self.RT_laser.copy()

        # xyz_image = compute_xyz(depth_image, self.fx, self.fy, self.px, self.py, self.height, self.width)
        # xyz_array = xyz_image.reshape((-1, 3))
        # xyz_base = np.matmul(RT_camera[:3, :3], xyz_array.T) + RT_camera[:3, 3].reshape(3, 1)
        # xyz_base = xyz_base.T.reshape((self.height, self.width, 3))
        return im_color, depth_image, RT_camera, RT_laser




def test_basic_img():
    # image listener
    rospy.init_node("image_listener")    
    listener = ImageListener()
    while 1:
        im_color, depth_image, xyz_image, xyz_base, RT_camera, intrinsics = listener.get_data()
        if im_color is None:
            continue
        print(f"cam {RT_camera}")
        # visualization
        fig = plt.figure()
    
        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(im_color)
    
        ax = fig.add_subplot(2, 2, 2)
        plt.imshow(depth_image)
    
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        pc = xyz_image.reshape((-1, 3))
        index = np.isfinite(pc[:, 2])
        pc = pc[index, :]
        n = pc.shape[0]
        index = np.random.choice(n, 5000)        
        ax.scatter(pc[index, 0], pc[index, 1], pc[index, 2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('sampled point cloud in camera frame')
        
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        pc = xyz_base.reshape((-1, 3))
        index = np.isfinite(pc[:, 2])
        pc = pc[index, :]
        n = pc.shape[0]
        index = np.random.choice(n, 5000)        
        ax.scatter(pc[index, 0], pc[index, 1], pc[index, 2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('sampled point cloud in base frame')
        plt.show()




if __name__ == '__main__':
    test_basic_img()
    
