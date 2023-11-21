import os.path
import json
import argparse
import numpy as np
import random
import datetime as dt
import copy
from sklearn.model_selection import train_test_split
import dataset as dsutil

parser = argparse.ArgumentParser(description='User args')
parser.add_argument('--dataset_dir', required=True, help='Path to dataset annotations')
parser.add_argument('--test_percentage', type=int, default=10, required=False, help='Percentage of images used for the testing set')
parser.add_argument('--val_percentage', type=int, default=10, required=False, help='Percentage of images used for the validation set')
parser.add_argument('--nr_trials', type=int, default=10, required=False, help='Number of splits')
parser.add_argument('--keep_categories', type=list, default=None, required=False, help="Super categories to keep distinguished")
parser.add_argument('--use_unofficial', type=Bool, default=False, required=False, help="boolean to indicate whether to use the unofficial TACO")
parser.add_argument('--seed', type=int, default=42, required=False, help="seed for the dataset splitting")


args = parser.parse_args()

add_on = "unofficial_" if args.use_unofficial else ""
ann_input_path = args.dataset_dir + '/' + add_on +'annotations.json'

# Load annotations
with open(ann_input_path, 'r') as f:
    dataset = json.loads(f.read())

if args.keep_categories is not None:
  class_map = dsutil.Taco.create_map(dataset["categories"], keep_categories)
  dsutil.Taco.replace_dataset_classes(dataset, class_map)

anns = dataset['annotations']
scene_anns = dataset['scene_annotations']
imgs = dataset['images']


for i in range(args.nr_trials):
    # Add new datasets
    train_set = {
        'info': None,
        'images': [],
        'annotations': [],
        'scene_annotations': [],
        'licenses': [],
        'categories': [],
        'scene_categories': [],
    }
    train_set['info'] =  dataset['info']
    train_set['categories'] = dataset['categories']
    train_set['scene_categories'] = dataset['scene_categories']

    val_set = copy.deepcopy(train_set)
    test_set = copy.deepcopy(train_set)

    train_set['images'], partial = train_test_split(dataset['images'],
                                                    random_state=args.seed+i,
                                                    test_size=args.test_percentage+args.val_percentage)
    val_set['images'], test_set["images"] = train_test_split(partial,
                                            random_state=args.seed+i,
                                            test_size=args.test_percentage/(args.test_percentage+args.val_percentage))

    # Aux Image Ids to split annotations
    test_img_ids, val_img_ids, train_img_ids = [],[],[]
    for img in test_set['images']:
        test_img_ids.append(img['id'])

    for img in val_set['images']:
        val_img_ids.append(img['id'])

    for img in train_set['images']:
        train_img_ids.append(img['id'])

    # Split instance annotations
    for ann in anns:
        if ann['image_id'] in test_img_ids:
            test_set['annotations'].append(ann)
        elif ann['image_id'] in val_img_ids:
            val_set['annotations'].append(ann)
        elif ann['image_id'] in train_img_ids:
            train_set['annotations'].append(ann)

    # Split scene tags
    for ann in scene_anns:
        if ann['image_id'] in test_img_ids:
            test_set['scene_annotations'].append(ann)
        elif ann['image_id'] in val_img_ids:
            val_set['scene_annotations'].append(ann)
        elif ann['image_id'] in train_img_ids:
            train_set['scene_annotations'].append(ann)

    # Write dataset splits
    ann_train_out_path = args.dataset_dir + '/' + 'annotations_' + str(i) +'_train.json'
    ann_val_out_path   = args.dataset_dir + '/' + 'annotations_' + str(i) + '_val.json'
    ann_test_out_path  = args.dataset_dir + '/' + 'annotations_' + str(i) + '_test.json'

    with open(ann_train_out_path, 'w+') as f:
        f.write(json.dumps(train_set))

    with open(ann_val_out_path, 'w+') as f:
        f.write(json.dumps(val_set))

    with open(ann_test_out_path, 'w+') as f:
        f.write(json.dumps(test_set))


