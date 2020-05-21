import io, os
from typing import Dict, Any

import tensorflow as tf
import ray, json
from xml.etree import ElementTree as ET

from COCO.COCO_2017_detection import build_tf_records

NUM_TRAIN_SHARDS = 2
NUM_VAL_SHARDS = 2
NUM_TEST_SHARDS = 2
ray.init()
tf.get_logger().setLevel('ERROR')


def download_coco():
    for d in ["val2017", "train2017", "test2017"]:
        if not os.path.exists(d):
            os.mkdir(d)

    os.system("wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar && \
    tar -xf VOCtrainval_06-Nov-2007.tar && \
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar && \
    tar -xf VOCtest_06-Nov-2007.tar &&\
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar && \
    tar -xf VOCtestnoimgs_06-Nov-2007.tar")


def parse_one_xml(xml_file: str, names_map: Dict) -> Dict[str, Any]:
    tree = ET.parse(os.path.join('./VOCdevkit/VOC2007/Annotations', xml_file))
    root = tree.getroot()
    filename = root.find('.//filename').text
    filepath = os.path.join('./VOCdevkit/VOC2007/JPEGImages', filename)
    objects_els = root.findall('.//object')
    size_el = root.find('size')
    width = int(size_el.find('width').text)
    height = int(size_el.find('height').text)
    depth = int(size_el.find('depth').text)

    bboxes = []
    for obj_el in objects_els:
        name_el = obj_el.find('name')
        bbox_el = obj_el.find('bndbox')
        bboxes.append({
            'class_text': name_el.text,
            'class_id': names_map[name_el.text],
            'xmin': int(bbox_el.find('xmin').text),
            'ymin': int(bbox_el.find('ymin').text),
            'xmax': int(bbox_el.find('xmax').text),
            'ymax': int(bbox_el.find('ymax').text),
        })

    return {
        'filepath': filepath,
        'filename': filename,
        'width': width,
        'height': height,
        'depth': depth,
        'bboxes': bboxes,
    }


def preprocess_voc(annot_dir: str) -> None:

    if not annot_dir == os.getcwd():
        os.chdir(annot_dir)

    print('Start to parse annotations.')

    if not os.path.exists(os.path.join(annot_dir, "tfrecords_voc")):
        os.makedirs(annot_dir + './tfrecords_voc')

    train_val_split = {}
    with open('./VOCdevkit/VOC2007/ImageSets/Main/train.txt') as train_fp:
        lines = train_fp.read().splitlines()
        for line in lines:
            train_val_split[line] = 'train'
    with open('./VOCdevkit/VOC2007/ImageSets/Main/val.txt') as val_fp:
        lines = val_fp.read().splitlines()
        for line in lines:
            train_val_split[line] = 'val'
    with open('./VOCdevkit/VOC2007/ImageSets/Main/test.txt') as val_fp:
        lines = val_fp.read().splitlines()
        for line in lines:
            train_val_split[line] = 'test'

    names = [
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    ]

    names_map = {name: i for i, name in enumerate(names)}
    print(names_map)
    train_annotations = []
    val_annotations = []
    test_annotations = []

    for xml_file in os.listdir('./VOCdevkit/VOC2007/Annotations'):
        image_id = xml_file[:-4]
        if train_val_split[image_id] == 'train':
            train_annotations.append(parse_one_xml(xml_file, names_map))
        elif train_val_split[image_id] == 'val':
            val_annotations.append(parse_one_xml(xml_file, names_map))
        elif train_val_split[image_id] == 'test':
            test_annotations.append(parse_one_xml(xml_file, names_map))
        else:
            print('WARNING: Unwanted image id {}'.format(image_id))
    print('Start to build TF Records.')
    build_tf_records(train_annotations, NUM_TRAIN_SHARDS, 'train')
    build_tf_records(val_annotations, NUM_VAL_SHARDS, 'val')
    build_tf_records(test_annotations, NUM_TEST_SHARDS, 'test')
    print('Successfully wrote {} annotations to TF Records.'.format(
        len(train_annotations) + len(val_annotations) + len(test_annotations)))
