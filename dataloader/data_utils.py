import pandas as pd
import json
import os
import numpy as np
import torch


def load_trainval_annotations(base_path=None):
    """
    load train and validation annotations from annotation files
    :return:
    :rtype:
    """
    if base_path is None:
        base_path = "D:\\data\\cvpr_birds\\annotation\\annotation"
    annots_files = {
        "train":      "anno_l_train.json",
        "validation": "anno_val.json",
    }
    # load train annotation file
    with open(os.path.join(base_path, annots_files["train"]), "r", encoding="ISO-8859-1") as file:
        train_annotations = json.load(file)
    with open(os.path.join(base_path, annots_files["validation"]), "r", encoding="ISO-8859-1") as file:
        valid_annotations = json.load(file)
    # get image details: each image detail and its annotation and merge them in one df
    train_imgs = pd.DataFrame(train_annotations['images'])
    train_anns = pd.DataFrame(train_annotations['annotations']).drop(columns='image_id')
    train_df = train_imgs.merge(train_anns, on='id')
    # for validation images
    valid_imgs = pd.DataFrame(valid_annotations['images'])
    valid_anns = pd.DataFrame(valid_annotations['annotations']).drop(columns='image_id')
    valid_df = valid_imgs.merge(valid_anns, on='id')

    return train_df, valid_df


def get_class_num():
    with open("/root/volume/birds/annotation/annotation/anno_l_train.json", "r", encoding="ISO-8859-1") as file:
        train = json.load(file)
    train_img = pd.DataFrame(train['images'])
    train_ann = pd.DataFrame(train['annotations']).drop(columns='image_id')
    train_df = train_img.merge(train_ann, on='id')
    # train_df = pd.DataFrame(train)
    a = train_df['category_id'].value_counts().to_dict()
    cat = []
    nums = []
    for i in range(200):
        nums.append(a[i])
        cat.append(i)
    return nums


def get_class_weights(beta):
    """  Taken from here: (https://arxiv.org/pdf/1906.07413.pdf)"""
    samples_per_cls = get_class_num()
    cls_num_list = samples_per_cls/np.sum(samples_per_cls)
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    return cls_num_list, per_cls_weights