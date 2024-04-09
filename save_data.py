from image_listener import ImageListener
import rospy
import numpy as np
from PIL import Image

class saveData:
    def __init__(self):
        rospy.init_node("img_listen_n_infer")
        self.listener = ImageListener("Fetch")
        self.time_delay = 3

    def save_data(self):
        data_count = 0
        while not rospy.is_shutdown():
            rgb, depth, RT_camera, RT_laser, RT_robot = self.listener.get_data_to_save()
            np.savez(f"{data_count}.npz", RT_camera=RT_camera, RT_robot=RT_robot)
            rgb = Image.fromarray(rgb)
            rgb.save(f"{data_count}.png")
            rospy.sleep(self.time_delay)
            data_count += 1
        
