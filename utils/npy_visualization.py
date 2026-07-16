import os
import numpy as np
from PIL import Image
import numpy as np
import torch
import torchvision

# prefix = r"E:\PycharmProjects\GhosetNetV3\GEGD\train\w_mask"
prefix = r"E:\PycharmProjects\GDD\test\gedfusion"
# RGB_prefix = r"E:\PycharmProjects\GhosetNetV3\GEGD\train\image"
RGB_prefix = r"E:\PycharmProjects\GDD\test\image"
# list = os.listdir(r"E:\PycharmProjects\GhosetNetV3\GEGD\train\w_mask")
list = os.listdir(r"E:\PycharmProjects\GDD\test\gedfusion")
for i in list:
    if i != "334.npy":
        continue

    path = os.path.join(prefix, i)
    raw_data = np.load(path)
    if np.max(raw_data) == 0:
        continue

    ged_fea = torch.tensor(raw_data)
    ged_fea_img = torch.nn.functional.sigmoid(ged_fea)
    # new_data = 1.0 * raw_data / np.max(raw_data) * 255
    # new_img = Image.fromarray(new_data)
    # torchvision.transforms.ToPILImage()(torch.chunk(ged_fea_img, 4, dim=0)[0]).show()
    new_img = Image.fromarray(ged_fea_img)

    new_img.show()
    print(i)
    RGB_name = i[:-4] + ".jpg"
    RGB_path = os.path.join(RGB_prefix, RGB_name)
    Image.open(RGB_path).show()
