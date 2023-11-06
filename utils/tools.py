from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import shutil
import yaml
import pandas as pd
from glob import glob

def boxes_to_coor(box):
    l = int(round(float(box[0])))
    t = int(round(float(box[1])))
    r = int(round(float(box[2])))
    b = int(round(float(box[3])))
    return l, t, r, b

def compute_normalized_cross_correlation(image1, image2):
    image1 = image1.squeeze(0)
    image2 = image2.squeeze(0)
    image1 = torch.permute(image1, (1, 2, 0))
    image2 = torch.permute(image2, (1, 2, 0))
    image1 = image1.detach().cpu().numpy()
    image2 = image2.detach().cpu().numpy()
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray1 = gray1.astype(np.float32)
    gray2 = gray2.astype(np.float32)
    mean1 = np.mean(gray1)
    mean2 = np.mean(gray2)
    gray1 -= mean1
    gray2 -= mean2
    std1 = np.std(gray1)
    std2 = np.std(gray2)
    ncc = (np.sum(gray1 * gray2) / (std1 * std2 * gray1.size)) - 1e-6
    return ncc 

def delete_min_object_iter(folder_result, name_img, num_object_min, num_iter_min):
    path_data = f'./img_results/{folder_result}/images_full/val/*'
    data_lst = sorted(glob(path_data))

    full_lst = []
    for img in data_lst:
        
        
        if(name_img[:-4] == os.path.split(img)[-1].split('_')[0]) and os.path.split(img)[-1].split('_')[1] != 'base.jpg':
            
            if (int(os.path.split(img)[-1].split('_')[1]) == num_object_min and int(os.path.split(img)[-1].split('_')[2][:-4]) == num_iter_min):
                continue
            full_lst.append(img)

        

    for path_img in full_lst:
        os.remove(path_img)

def delete_temp_img(folder_result):
    path = f'./img_results/{folder_result}/images_full/val/*_*_*.jpg'
    path_mean = f'./img_results/{folder_result}/images_full/val/*_ncc_mean_*.jpg'
    base_path = sorted(glob(path))
    mean_path = sorted(glob(path_mean))
    for path_img in mean_path:
        base_path.remove(path_img)
    for path_img in base_path:
        os.remove(path_img)

def save_loss(loss_log_image, name_of_fol, name_image):
    path = f'./img_results/{name_of_fol}'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
    path = f'./img_results/{name_of_fol}/loss'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
    
    #np.savetxt(f'{name_image}_loss.txt',torch.tensor(loss_log_image).detach().cpu().numpy(), fmt = '%.18f')
    plt.plot(torch.tensor(loss_log_image).detach().cpu().numpy())
    plt.savefig(f'./img_results/{name_of_fol}/loss/{name_image}')
    plt.clf()
    plt.close()

def save_visualize_image(img, name_of_fol ,name, loss_value):
    path = f'./img_results/{name_of_fol}'
    isExist = os.path.exists(path)
    if isExist == False:
        os.makedirs(path, exist_ok=True)
    path = f'./img_results/{name_of_fol}/images_full'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.makedirs(path, exist_ok=True)
    path = f'./img_results/{name_of_fol}/images_full/val'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.makedirs(path, exist_ok=True)
    save_image(img, f'./img_results/{name_of_fol}/images_full/val/{name[:-4]}_{loss_value}{name[-4:]}')
    path = f'./img_results/{name_of_fol}/labels'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.makedirs(path, exist_ok=True)
    path = f'./img_results/{name_of_fol}/labels/val/'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.makedirs(path, exist_ok=True)
 
    #shutil.copy(f'./datasets/MOT20Det/train/MOT20-01/output/{name[:-4]}.txt',path + f'{name[:-4]}_{loss_value}.txt')

