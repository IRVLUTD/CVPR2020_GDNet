#!/usr/bin/python3
import logging
import os
import time

import numpy as np
import rospkg
import rospy
import tf
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory
from control_msgs.msg import GripperCommand
from transforms3d import quaternions
from quaternions import quat2mat
from transforms3d.euler import euler2quat
import tf2_ros
from sensor_msgs.msg import LaserScan
from image_listener import ImageListener

class SimNode:
    def __init__(self):
        rospy.init_node("igibson_sim")
        

        self.base_frame = "base_link"
        self.camera_frame = "head_camera_rgb_optical_frame"

        
        # self.depth_pub = rospy.Publisher("/gibson_ros/camera/depth/image", ImageMsg, queue_size=10)
        # self.lidar_pub = rospy.Publisher("/gibson_ros/lidar/points", PointCloud2, queue_size=10)
        # self.depth_raw_pub = rospy.Publisher("/gibson_ros/camera/depth/image_raw", ImageMsg, queue_size=10)
        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=10)

        #Publisher for depth image transformed into laser link frame
        self.listener = ImageListener("Gazebo")

        self.br = tf.TransformBroadcaster()

        
        self.tp_time = None

        #Camera parameters  
        self.height=480
        self.width=640
        self.indices = np.indices((self.height, self.width), dtype=np.float32).transpose(1,2,0)
        self.intrinsics = [[554.254691191187, 0.0, 320.5],[0.0, 554.254691191187, 240.5], [0.0, 0.0, 1.0]]
        self.fx = self.intrinsics[0][0]
        self.fy = self.intrinsics[1][1]
        self.px = self.intrinsics[0][2]
        self.py = self.intrinsics[1][2]

        #Camera pose
        self.eye_pos = np.array([0,0,0])
        self.eye_orn = np.array([0,0,0,0])
        self.eye_pose_message = Pose()

        #Depth Image initialsiation
        self.depth_image = None


        #Base link to laser link transform check
        self.transform_received = False
        self.laserscan = None
        self.revised_scan = LaserScan()
        self.laser_pub =  rospy.Publisher('/revised_scan', LaserScan, queue_size = 10)
        rospy.Subscriber("/scan", LaserScan, self.laserscan_callback)

        #Tf listener to get static transform between base link and laser link
        self.tfBuffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        #Parameters to clip pointcloud at specified height range
        self.max_height = rospy.get_param("~height_max_cutoff")
        self.min_height = rospy.get_param("~height_min_cutoff")

        self.depth_sampling_frequency = 1/60
        
        self.num_levels = rospy.get_param("~num_levels")
        self.step_size = rospy.get_param("~step_size")

    def laserscan_callback(self, scan):
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
        

    def xyzw2wxyz(self, orn):
        """
        :param orn: quaternion in xyzw
        :return: quaternion in wxyz
        """
        return [orn[-1], orn[0], orn[1], orn[2]]

    def quat2rotmat(self, quat):
        """
        :param quat: quaternion in w,x,y,z
        :return: rotation matrix 4x4
        """
        rot_mat = np.eye(4)
        rot_mat[:3, :3] = quaternions.quat2mat(quat)
        return rot_mat

    def quat2rotmat(self, quat):
        """
        :param quat: quaternion in w,x,y,z
        :return: rotation matrix 4x4
        """
        rot_mat = np.eye(4)
        rot_mat[:3, :3] = quaternions.quat2mat(quat)
        return rot_mat

    def compute_xyz(self, depth_image):
        '''
        Function to convert Depth image into 4D pointcloud. 4th D being 1s for homogeneous coordinates
        '''
        depth_image = np.array(depth_image)
        indices = self.indices
        depth_image = depth_image.reshape(self.height, self.width)
        homogenious = np.ones([self.height, self.width])
        x_e = (indices[..., 1] - self.px) * depth_image / self.fx
        y_e = (indices[..., 0] - self.py) * depth_image/ self.fy
        xyz_img = np.stack([x_e, y_e, depth_image, homogenious], axis=-1) # Shape: [H x W x 4]
        return xyz_img

    def get_depth_map(self, laser_trans):


        '''
        Convert Depth point cloud from robot camera frame to laser link frame
        '''
        #Get Base Link pose
        # self.base_position, self.base_orientation = self.env.robots[0].base_link.get_position_orientation()
        
        #Get Base Link to Laser Link transformation
        laser_rotation = quat2mat([ laser_trans.transform.rotation.w, laser_trans.transform.rotation.x, laser_trans.transform.rotation.y, laser_trans.transform.rotation.z])
        laser_translation = [laser_trans.transform.translation.x, laser_trans.transform.translation.y, laser_trans.transform.translation.z]

        # Get Camera Pose
        # self.eye_pos, self.eye_orn = self.env.robots[0].links["eyes"].get_position_orientation()

        #Convert Camera pose into world frame
        camera_in_wf = self.quat2rotmat(self.xyzw2wxyz(self.eye_orn))
        camera_in_wf[:3,3] = self.eye_pos

        #Pose of the simulated robot in world frame
        robot_pos, robot_orn = self.env.robots[0].get_position_orientation()
        robot_in_wf = self.quat2rotmat(self.xyzw2wxyz(robot_orn))
        robot_in_wf[:3, 3] = robot_pos        

        #Pose of the camera in robot frame
        cam_in_robot_frame = np.dot(np.linalg.inv(robot_in_wf), camera_in_wf)

        #3D image from igibson simulation
        [td_image] = self.env.simulator.renderer.render(modes=('3d'))
        td_image = np.transpose(td_image, (2, 0, 1))
        td_image = td_image.reshape((4, -1))
        
        #Transforming coordinates of points from opengl frame to camera frame
        camera_in_openglf = self.quat2rotmat(euler2quat(np.pi / 2.0, 0, -np.pi / 2.0))

        #Convert points in camera frame to world frame
        point_in_cf = np.matmul(camera_in_openglf, td_image)
        point_in_rf = np.dot(cam_in_robot_frame, point_in_cf)
        point_in_wf = np.dot(robot_in_wf, point_in_rf).T

        #Convert points in world frame to base link frame
        base_translation = self.base_position
        base_rotation = quat2mat(
            [self.base_orientation[3], self.base_orientation[0], self.base_orientation[1], self.base_orientation[2]]
        )
        points_local = base_rotation.T.dot((point_in_wf[:, :3] - base_translation).T).T

        #Convert points in base link frame to laser link frame
        points_local = laser_rotation.T.dot((points_local - laser_translation).T).T

        #After convertion filter out values outside the range (multilevel) and downsample the resulting depth image
        points = None
        for level in range(self.num_levels):
            index =  (points_local[:,2] > (-self.max_height - level * self.step_size)) & (points_local[:,2] < (-self.min_height - level * self.step_size))
            if points is None:
                points = points_local[index,:]
            else:
                points = np.concatenate((points, points_local[index,:]), axis=0)
        points_local = points
        points_local = points_local[0:points_local.size:int(1/self.depth_sampling_frequency)]

        return points_local


    def run(self):
        
        action = np.zeros(self.robot.action_dim)
        lidar_header = Header()
        pc_header = Header()

        while not self.transform_received:
            try:
                trans = self.tfBuffer.lookup_transform('base_link', 'laser_link', rospy.Time())
                self.transform_received = True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as error:

                continue


        # laser_linear_range = self.laserscan["range_max"] - self.laserscan["range_min"]
        # laser_angular_range = self.laserscan["angle_max"] - self.laserscan["angle_min"]
        # min_laser_dist = 0.3
        # n_horizontal_rays = 228

        # laser_angular_half_range = laser_angular_range / 2.0
        # angle = np.arange(
        #     -np.radians(laser_angular_half_range),
        #     np.radians(laser_angular_half_range),
        #     np.radians(laser_angular_range) / n_horizontal_rays,
        # )
        # unit_vector_laser = np.array([[np.cos(ang), np.sin(ang), 0.0] for ang in angle])

    

        rate = rospy.Rate(15)

        while not rospy.is_shutdown():


            
            rgb = (obs["rgb"] * 255).astype(np.uint8) #get from image listreneer
            # normalized_depth = obs["depth"].astype(np.float32)
            # depth = normalized_depth * self.env.sensors["vision"].depth_high
            # self.depth_image = depth
            # depth_raw_image = (obs["depth"] * 1000).astype(np.uint16)

        




            self.eye_pos, self.eye_orn = self.env.robots[0].links["eyes"].get_position_orientation()  # get from rt camera image listener
            self.eye_pose_message.header.stamp = rospy.Time.now()
            self.eye_pose_message.header.frame_id = "head_camera_depth_optical_frame"
            self.eye_pose_message.position.x = self.eye_pos[0]
            self.eye_pose_message.position.y = self.eye_pos[1]
            self.eye_pose_message.position.z = self.eye_pos[2]
            self.eye_pose_message.orientation.x = self.eye_orn[0]
            self.eye_pose_message.orientation.y = self.eye_orn[1]
            self.eye_pose_message.orientation.z = self.eye_orn[2]
            self.eye_pose_message.orientation.w = self.eye_orn[3]

            self.eye_pub.publish(self.eye_pose_message)

            if (self.tp_time is None) or (
                (self.tp_time is not None) and ((rospy.Time.now() - self.tp_time).to_sec() > 1.0)
            ):

                depth_points = self.get_depth_map(trans)

                '''
                Publish the depth pointcloud. Uncomment below to publish
                '''
                # pc_header.stamp = rospy.Time.now()
                # pc_header.frame_id = "laser_link"
                # pc_message = pc2.create_cloud_xyz32(pc_header, depth_points)
                # self.pc_pub.publish(pc_message)

                scan = self.laserscan["ranges"]
                lidar_header.stamp = rospy.Time.now()
                lidar_header.frame_id = "laser_link"

                
                lidar_points = unit_vector_laser * (scan * (laser_linear_range - min_laser_dist) + min_laser_dist)

                #Setting height to zero since we need to project scan at o level ( lidar pov)
                depth_points[:,2] = 0

                #merging lidar points and depth points together

                lidar_points = np.concatenate((lidar_points, depth_points), axis=0)

                #converting the points into pointcloud format
                lidar_message = pc2.create_cloud_xyz32(lidar_header, lidar_points)

                self.lidar_pub.publish(lidar_message)

                

                rate.sleep()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    node = SimNode()
    node.run()
