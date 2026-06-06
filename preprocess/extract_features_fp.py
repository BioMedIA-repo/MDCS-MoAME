import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from math import floor
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
from einops import rearrange, repeat
import cv2
from models.builder import get_encoder
def vertical_scanning_tensor(input_tensor):

    output_tensor = rearrange(input_tensor, 'b p1 p2 c -> b p2 p1 c ')
    output_tensor = rearrange(output_tensor, 'b p1 p2 c -> b (p2 p1) c ')

    return output_tensor

def left_oblique_scanning_tensor(input_tensor):

    batch_size, p1, p2, channels = input_tensor.shape

    output_tensor = torch.empty(batch_size, p1 * p2, channels)

    for b in range(batch_size):
        output_tensor[b, 0, :] = input_tensor[b, 0, 0, :]
        x_len = p2 - 1
        y_len = p1 - 1
        i = 0
        j = 0
        t = 1
        while (1):
            if (i < x_len):
                i = i + 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1
                if (i == x_len and j == y_len):
                    break
            elif (i == x_len and j < y_len):
                j = j + 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1
                if (i == x_len and j == y_len):
                    break
            while (i > 0 and j < y_len):
                i = i - 1
                j = j + 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1
            if (j < y_len):
                j = j + 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1
                if (i == x_len and j == y_len):
                    break
            elif (j == y_len and i < x_len):
                i = i + 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1
                if (i == x_len and j == y_len):
                    break
            while (j > 0 and i < x_len):
                j = j - 1
                i = i + 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1
    return output_tensor

def right_oblique_scanning_tensor(input_tensor):

    batch_size, p1, p2, channels = input_tensor.shape

    output_tensor = torch.empty(batch_size, p1 * p2, channels)

    for b in range(batch_size):
        x_len = p2 - 1
        y_len = p1 - 1
        output_tensor[b, 0, :] = input_tensor[b, 0, x_len, :]
        i = x_len
        j = 0
        t = 1
        while(1):
            if(i > 0):
                i = i - 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1
                if (i == 0 and j == y_len):
                    break
            elif(i == 0 and j < y_len):
                j = j + 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1
                if (i == 0 and j == y_len):
                    break
            while(i < x_len and j < y_len):
                i = i + 1
                j = j + 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1
            if(j < y_len):
                j = j + 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1
                if (i == 0 and j == y_len):
                    break
            elif(j == y_len and i > 0):
                i = i - 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1
                if (i == 0 and j == y_len):
                    break
            while(j > 0 and i > 0):
                j = j - 1
                i = i - 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1

    return output_tensor

def loopback_scanning_tensor(input_tensor):

    batch_size, p1, p2, channels = input_tensor.shape

    output_tensor = torch.empty(batch_size, p1 * p2, channels)

    for b in range(batch_size):
        x_len = p2 - 1
        y_len = p1 - 1
        x_left = 0
        y_left = 0
        output_tensor[b, 0, :] = input_tensor[b, 0, 0, :]

        i = 0
        j = 0
        t = 1
        index = int((x_len + 1) * (y_len + 1) - 1)
        point = 0
        y_left = y_left + 1
        while (1):
            if (point == index):
                break
            while (i < x_len):
                i = i + 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1
                point = point + 1
            x_len = x_len - 1
            if (point == index):
                break
            while (j < y_len):
                j = j + 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1
                point = point + 1
            y_len = y_len - 1
            if (point == index):
                break
            while (i > x_left):
                i = i - 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1
                point = point + 1
            x_left = x_left + 1
            if (point == index):
                break
            while (j > y_left):
                j = j - 1
                output_tensor[b, t, :] = input_tensor[b, j, i, :]
                t = t + 1
                point = point + 1
            y_left = y_left + 1
            if (point == index):
                break

    return output_tensor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default='/data/path/to/TCGA_BLCA/patch')
parser.add_argument('--data_slide_dir', type=str, default='/data/path/to/TCGA_BLCA/svs')
parser.add_argument('--csv_path', type=str, default='/data/path/to/TCGA_BLCA/patch/process_list_autogen.csv')
parser.add_argument('--feat_dir', type=str, default='/data/path/to/TCGA_BLCA/pt')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
args = parser.parse_args()

def compute_w_loader(file_path, output_path, wsi, model,
    batch_size = 8, verbose = 0, print_every=20, pretrained=True,
    custom_downsample=1, target_patch_size=-1):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained,
        custom_downsample=custom_downsample, target_patch_size=target_patch_size)
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path,len(loader)))

    mode = 'w'
    for count, (batch, batch2, batch3, batch4, batch5, batch6, batch7, batch8, batch9, batch10, coords, coords_vec, coord_topleft, coord_topright, coord_hui) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            # import ipdb;
            # ipdb.set_trace()
            batch = batch.to(device, non_blocking=True) # (4,3,4096,4096)
            batch2 = batch2.to(device, non_blocking=True)
            batch3 = batch3.to(device, non_blocking=True) # (4,3,4096,4096)
            batch4 = batch4.to(device, non_blocking=True)
            batch5 = batch5.to(device, non_blocking=True)  # (4,3,4096,4096)
            batch6 = batch6.to(device, non_blocking=True)  # (4,3,4096,4096)
            batch7 = batch7.to(device, non_blocking=True)  # (4,3,4096,4096)
            batch8 = batch8.to(device, non_blocking=True)  # (4,3,4096,4096)
            batch9 = batch9.to(device, non_blocking=True)  # (4,3,4096,4096)
            batch10 = batch10.to(device, non_blocking=True)  # (4,3,4096,4096)
            x = int(batch.shape[2] / 8)
            y = int(batch.shape[3] / 8)
            batch = batch.unfold(2, x, x).unfold(3, y, y)
            # print(batch.shape)
            batch = rearrange(batch, 'b c p1 p2 w h -> (b p1 p2) c w h') # (256,3,512,512)
            batch3 = batch3.unfold(2, x, x).unfold(3, y, y)
            batch3 = rearrange(batch3, 'b c p1 p2 w h -> (b p1 p2) c w h') # (256,3,512,512)
            batch5 = batch5.unfold(2, x, x).unfold(3, y, y)
            batch5 = rearrange(batch5, 'b c p1 p2 w h -> (b p1 p2) c w h') # (256,3,512,512)
            batch7 = batch7.unfold(2, x, x).unfold(3, y, y)
            batch7 = rearrange(batch7, 'b c p1 p2 w h -> (b p1 p2) c w h') # (256,3,512,512)
            batch9 = batch9.unfold(2, x, x).unfold(3, y, y)
            batch9 = rearrange(batch9, 'b c p1 p2 w h -> (b p1 p2) c w h') # (256,3,512,512)

            features = model(batch)
            features2 = model(batch2)
            features3 = model(batch3)
            features4 = model(batch4)
            features5 = model(batch5)
            features6 = model(batch6)
            features7 = model(batch7)
            features8 = model(batch8)
            features9 = model(batch9)
            features10 = model(batch10)

            features = rearrange(features, '(b p1 p2) c -> b p1 p2 c', p1=8, p2=8) # (4,8,8,1024)
            features = features.cpu().numpy()
            features2 = features2.cpu().numpy() #(4,1024)

            features3 = rearrange(features3, '(b p1 p2) c -> b p1 p2 c', p1=8, p2=8) # (4,8,8,1024)
            features3 = vertical_scanning_tensor(features3)
            features3 = features3.cpu().numpy()
            features4 = features4.cpu().numpy() #(4,1024)

            features5 = rearrange(features5, '(b p1 p2) c -> b p1 p2 c', p1=8, p2=8) # (4,8,8,1024)
            features5 = left_oblique_scanning_tensor(features5)
            features5 = features5.cpu().numpy()
            features6 = features6.cpu().numpy() #(4,1024)

            features7 = rearrange(features7, '(b p1 p2) c -> b p1 p2 c', p1=8, p2=8) # (4,8,8,1024)
            features7 = right_oblique_scanning_tensor(features7)
            features7 = features7.cpu().numpy()
            features8 = features8.cpu().numpy() #(4,1024)

            features9 = rearrange(features9, '(b p1 p2) c -> b p1 p2 c', p1=8, p2=8) # (4,8,8,1024)
            features9 = loopback_scanning_tensor(features9)
            features9 = features9.cpu().numpy()
            features10 = features10.cpu().numpy() #(4,1024)

            coords = np.array(coords)
            coords_vec = np.array(coords_vec)
            coord_topleft = np.array(coord_topleft)
            coord_topright = np.array(coord_topright)
            coord_hui = np.array(coord_hui)


            asset_dict = {'features': features, 'features2': features2, 'features3': features3, 'features4': features4, 'features5': features5,
                          'features6': features6, 'features7': features7, 'features8': features8, 'features9': features9, 'features10': features10,
                          'coords': coords, 'coords_vec': coords_vec, 'coord_topleft': coord_topleft, 'coord_topright': coord_topright, 'coord_hui': coord_hui}
            save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
            mode = 'a'

    return output_path

if __name__ == '__main__':

    print('initializing dataset')

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'h5_files'))

    print('loading model checkpoint1')

    # resnet50
    model, _ = get_encoder(args.model_name, target_img_size=args.target_patch_size)
    _ = model.eval()
    model = model.to(device)

    slides = os.listdir(args.data_h5_dir + '/patches')
    total = len(slides)

    for bag_candidate_idx in range(total):
        slide_id = slides[bag_candidate_idx].split('.h5')[0]
        bag_name = slide_id+'.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)
        if not args.no_auto_skip and slide_id + '.h5' in dest_files:
            print('skipped {}'.format(slide_id))
            continue

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        output_file_path = compute_w_loader(h5_file_path, output_path, wsi, model = model, batch_size = args.batch_size, verbose = 1, print_every = 20,
        custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
        file = h5py.File(output_file_path, "r")
