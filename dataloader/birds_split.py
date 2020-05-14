# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
import json
import pandas as pd
from shutil import copyfile
from data_utils import load_trainval_annotations


def cvpr_birds_trainval_split(source_folder, dest_folder):
    """
    train-val split for USC birds dataset
    :param folder: folder containing dataset
    :return: nothing, save results in train and val folders
    """
    # load train and valid information
    train_df, valid_df = load_trainval_annotations()

    # create folders for each class in val and train folder, to prepare for pytorch load
    if not os.path.exists(os.path.join(dest_folder, 'val')):
        for i in range(200):
            os.makedirs(os.path.join(dest_folder, 'val/'+str(i)))
    if not os.path.exists(os.path.join(dest_folder, 'train')):
        for i in range(200):
            os.makedirs(os.path.join(dest_folder, 'train/'+str(i)))

    # copy images from trainval folder to train and val folder
    for cls in tqdm(range(200)):
        # files for train and validation classes
        train_imgs_per_cls = list(train_df[train_df.category_id == cls]['file_name'].values)
        val_imgs_per_cls = list(valid_df[valid_df.category_id == cls]['file_name'].values)

        for img_path in train_imgs_per_cls:
            img_name = img_path.split('/')[-1]
            src_file_path = os.path.join(source_folder, img_path)
            dst_file_path = os.path.join(dest_folder, "train", str(cls), img_name)
            copyfile(src_file_path, dst_file_path)

        for img_path in val_imgs_per_cls:
            img_name = img_path.split('/')[-1]
            src_file_path = os.path.join(source_folder, img_path)
            dst_file_path = os.path.join(dest_folder, "val", str(cls), img_name)
            copyfile(src_file_path, dst_file_path)


if __name__ == "__main__":
    cvpr_birds_trainval_split("D:\\data\\cvpr_birds\\trainval_images", "D:\\data\\cvpr_birds")