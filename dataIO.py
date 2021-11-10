from PIL import Image
import numpy as np
import os
import torch

def get_image(img_path):
    img = Image.open(img_path)
    real_img = img.crop((0, 0, img.size[0] / 2, img.size[1]))
    cond_img = img.crop((img.size[0] / 2, 0, img.size[0], img.size[1]))
    real_img = np.array(real_img, dtype=np.float32) / 255. * 2. - 1.
    cond_img = np.array(cond_img, dtype=np.float32) / 255. * 2. - 1.
    return real_img, cond_img

def get_data_set(data_path, data_set_size, image_size):
    max_size = len(os.listdir(data_path))
    if data_set_size > max_size: data_set_size = max_size
    real_img_set = np.zeros((data_set_size, image_size[0], image_size[1], 3))
    cond_img_set = np.zeros((data_set_size, image_size[0], image_size[1], 3))
    for id in range(data_set_size):
        img_path = data_path+str(id+1)+'.jpg'
        real_img, cond_img = get_image(img_path)
        real_img_set[id] = real_img
        cond_img_set[id] = cond_img
    real_img_set = np.transpose(real_img_set, (0, 3, 1, 2))
    cond_img_set = np.transpose(cond_img_set, (0, 3, 1, 2))
    return real_img_set, cond_img_set

def val_to_img(img, model, is_tensor = True):
    if is_tensor : img = img.detach().numpy()
    else : img = np.array(img)
    img = np.transpose(img, (0, 2, 3, 1))
    img = np.reshape(img, (-1, 256, 256, 3))
    img = np.array((img + 1.) / 2. * 255., dtype=np.uint8)
    return img

def make_sample_img(model, test_cond_img, test_img, sample_size):
    sample_img = np.zeros((sample_size, 256, 768, 3), dtype=np.uint8)

    test_cond_img = torch.tensor(test_cond_img, dtype=torch.float32)
    generated_img = model.generator(test_cond_img)
    generated_img = val_to_img(generated_img, model)
    test_cond_img = val_to_img(test_cond_img, model)
    test_img = val_to_img(test_img, model, is_tensor=False)

    sample_img[:, :, :256, :] = test_cond_img
    sample_img[:, :, 256:512, :] = test_img
    sample_img[:, :, 512:768, :] = generated_img

    for id, sample in enumerate(sample_img):
        img = Image.fromarray(sample)
        img.save("./sample/sample"+str(id)+".jpeg")
    return 0





