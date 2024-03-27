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
        self.net = GDNet()
        self.kernel = np.ones((3,3),np.uint8)

    def main(self):
        rospy.init_node("img_listen_n_infer")
        self.listener = ImageListener("Gazebo")
        if len(self.args['snapshot']) > 0:
            print(f"checking snapshots\n")
            print('Load snapshot {} for testing'.format(self.args['snapshot']))
            self.net.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.exp_name, self.args['snapshot'] + '.pth')))
            print('Load {} succeed!'.format(os.path.join(self.ckpt_path, self.exp_name, self.args['snapshot'] + '.pth')))
            self.net.eval()
            with torch.no_grad():
                while not rospy.is_shutdown():
                    rgb, depth, _, _ = self.listener.get_data()
                    rgb = Image.fromarray(rgb)
                    w,h = rgb.size
                    img_var = Variable(self.img_transform(rgb).unsqueeze(0))
                    f1, f2, f3 = self.net(img_var)
                    f3 = f3.data.squeeze(0).cpu()
                    f3 = np.array(transforms.Resize((h, w))(self.to_pil(f3)))
                    if self.args['crf']:
                        f3 = crf_refine(np.array(rgb.convert('RGB')), f3)
                    print(f"type of f3 {type(f3)}")
                    non_z_depth = np.where(depth > 0)
                    f3[non_z_depth] = 0
                    f3 = cv2.morphologyEx(f3, cv2.MORPH_CLOSE, self.kernel)
                    rospy.sleep(1)


if __name__ == '__main__':
    infy = Inference()
    infy.main()
    rospy.spin()