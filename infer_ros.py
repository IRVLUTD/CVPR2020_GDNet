"""
 @Time    : 2020/3/15 20:43
 @Author  : TaylorMei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2020_GDNet
 @File    : infer.py
 @Function:
 
"""
import os
import time

import numpy as np

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import gdd_testing_root, gdd_results_root
from misc import check_mkdir, crf_refine
from gdnet import GDNet

from image_listener import ImageListener
import rospy
import cv2
import ros_numpy
from grasp_utils import compute_xyz
from ros_utils import assign_depth_to_segments
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image as Img
from std_msgs.msg import Header

class Inference:
    def __init__(self) -> None:
        
        self.device_ids = [0]
        torch.cuda.set_device(self.device_ids[0])

        print(f"infer starts\n")
        self.ckpt_path = './ckpt'
        self.exp_name = 'GDNet'
        self.args = {
            'snapshot': '200',
            'scale': 416,
            # 'crf': True,
            'crf': False,
        }

        print(torch.__version__)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.args['scale'], self.args['scale'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.to_test = {'GDD': gdd_testing_root}

        self.to_pil = transforms.ToPILImage()
        ###############
        # self.net = GDNet()
        ###############
        self.net = GDNet().cuda(self.device_ids[0])

        self.kernel = np.ones((3,3),np.uint8)

    def main(self):
        rospy.init_node("img_listen_n_infer")
        self.listener = ImageListener("Fetch")
        self.segm_imgpub = rospy.Publisher("/segmented_img", Img, queue_size=5)

        rospy.sleep(2)
        self.unit_lidar_vector, range, min_dist = self.listener.get_unit_lidar_vector()
        self.lidar_header = Header()
        if len(self.args['snapshot']) > 0:
            print(f"checking snapshots\n")
            print('Load snapshot {} for testing'.format(self.args['snapshot']))
            self.net.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.exp_name, self.args['snapshot'] + '.pth')))
            print('Load {} succeed!'.format(os.path.join(self.ckpt_path, self.exp_name, self.args['snapshot'] + '.pth')))
            self.net.eval()
            with torch.no_grad():
                while not rospy.is_shutdown():
                    rgb, depth, RT_camera, RT_laser = self.listener.get_data()
                    # print(f"RT laser {RT_laser}")
                    rgb = Image.fromarray(rgb)
                    w,h = rgb.size
                    ######################
                    # img_var = Variable(self.img_transform(rgb).unsqueeze(0))
                    ######################
                    img_var = Variable(self.img_transform(rgb).unsqueeze(0)).cuda(self.device_ids[0])
                    f1, f2, f3 = self.net(img_var)
                    ######################
                    # f3 = f3.data.squeeze(0).cpu()
                    ######################
                    f3 = f3.data.squeeze(0)
                    f3 = np.array(transforms.Resize((h, w))(self.to_pil(f3)))
                    if self.args['crf']:
                        f3 = crf_refine(np.array(rgb.convert('RGB')), f3)
                    # print(f"type of f3 {type(f3)}")
                    non_z_depth = np.where(depth > 0)
                    f3[non_z_depth] = 0
                    f3 = cv2.morphologyEx(f3, cv2.MORPH_CLOSE, self.kernel)
                    depth = assign_depth_to_segments(f3, depth)
                    f4 = ros_numpy.msgify(Img, f3,encoding='mono8')
                    self.segm_imgpub.publish(f4)
                    #assumew we have the updated depth image
                    xyz_image = compute_xyz(depth, self.listener.fx,self.listener.fy,self.listener.px,self.listener.py, h, w )
                    # print(xyz_image.shape)
                    xyz_array = xyz_image.reshape((-1, 3))
                    # print(xyz_array.shape)
                    xyz_base = np.matmul(RT_camera[:3, :3], xyz_array.T) + RT_camera[:3, 3].reshape(3, 1)
                    xyz_base = xyz_base.reshape((-1, 3))

                    xyz_base = xyz_base[(xyz_base[:,2]<0.5) & (xyz_base[:,2]>0.3)]
                    xyz_base = xyz_base[0:xyz_base.size:30]
                    # done with points in base frame
                    # print(xyz_base.shape)
                    xyz_laser = np.matmul(RT_laser[:3,:3], xyz_base.T) + RT_laser[:3,3].reshape(3,1)
                    # xyz_laser = xyz_laser.T.reshape((h, w, 3))
                    xyz_laser = xyz_laser.T.reshape((-1,3))
                    ranges = self.listener.laserscan['ranges']
                    ranges = ranges[:, np.newaxis]
                    lidar_pc = self.unit_lidar_vector * ranges
                    xyz_laser[:,2] = 0

                    # print(f"lidar pc {lidar_pc.shape}\n")

                    lidar_pc = np.concatenate((lidar_pc, xyz_laser), axis=0)
                    self.lidar_header.stamp = rospy.Time.now()
                    self.lidar_header.frame_id = "laser_link"
                    lidar_message = pc2.create_cloud_xyz32(self.lidar_header, lidar_pc)
                    self.listener.lidar_pub.publish(lidar_message)

                    # done with points in laser frame
                    print(f"one iteration completed")
                    # rospy.sleep(1)


if __name__ == '__main__':
    infy = Inference()
    infy.main()
    rospy.spin()