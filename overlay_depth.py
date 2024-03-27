from PIL import Image
import numpy as np
import os
from PIL import ImageFilter

results_path = "/home/ash/irvl/CVPR2020_GDNet/results/GDNet_200/segmented/GDNet_200"
results = [img for img in os.listdir(results_path)]

depth_path = "/home/ash/irvl/CVPR2020_GDNet/results/GDNet_200/depth"
depths = [img for img in os.listdir(depth_path)]

refined_result_path = "/home/ash/irvl/CVPR2020_GDNet/results/GDNet_200/refined"

results.sort()
depths.sort()


# pseudo code
'''
1. read depth image
2. read result image
3. get where depth >0
4. remove segments of that region in result image
5. Perform Erosion operation
'''
for i in range(len(depths)):
    
    depth_img = Image.open(os.path.join( depth_path, depths[i]))
    result_img = Image.open(os.path.join( results_path, results[i]))

    # print(type(np.array(depth_img)))

    non_z_depth = np.where(np.array(depth_img) > 0)
    refined_result = np.array(result_img)
    refined_result[non_z_depth] = 0

    
    refined_result = Image.fromarray(refined_result)
    refined_result = refined_result.filter(ImageFilter.MinFilter(9))
    # refined_result = refined_result.filter(ImageFilter.MaxFilter(5))
    refined_result.save(os.path.join(refined_result_path, f"{results[i]}"))

    print(f"refined {i}th result")
