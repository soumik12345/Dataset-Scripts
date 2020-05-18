import csv, io, os
from typing import Dict, List

import json
import ray
import tensorflow as tf

from PIL import Image

NUM_TRAIN_SHARDS = 64
NUM_VAL_SHARDS = 8
ray.init()
tf.get_logger().setLevel("ERROR")


def _chunkify(data, n_chunks) -> List:
    size = len(data) // n_chunks
    start = 0
    results = []
    for i in range(n_chunks - 1):
        results.append(data[start:start + size])
        start += size
    results.append(data[start:])
    return results


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _generate_tf_example(anno_list) -> tf.train.Example:

    filename = anno_list[0]['filename']
    with open(filename, 'rb') as image_file:
        content = image_file.read()
    image = Image.open(filename)

    if image.format != 'JPEG' or image.mode != 'RGB':
        image_rgb = image.convert('RGB')
        with io.BytesIO() as output:
            image_rgb.save(output, format="JPEG", quality=95)
            content = output.getvalue()

    width, height = image.size
    depth = 3
    class_ids = []
    class_texts = []
    bbox_xmins = []
    bbox_ymins = []
    bbox_xmaxs = []
    bbox_ymaxs = []

    for anno in anno_list:
        class_ids.append(anno['class_id'])
        class_texts.append(anno['class_text'].encode())
        xmin, ymin, xmax, ymax = anno['xmin'], anno['ymin'], anno[
            'xmax'], anno['ymax']
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = float(
            xmin) / width, float(ymin) / height, float(xmax) / width, float(
                ymax) / height
        assert 1 >= bbox_xmin >= 0, "bbox_xmin is invalid, should be in [0,1]"
        assert 1 >= bbox_ymin >= 0, "bbox_ymin is invalid, should be in [0,1]"
        assert 1 >= bbox_xmax >= 0, "bbox_xmax is invalid, should be in [0,1]"
        assert 1 >= bbox_ymax >= 0, "bbox_ymax is invalid, should be in [0,1]"
        bbox_xmins.append(bbox_xmin)
        bbox_ymins.append(bbox_ymin)
        bbox_xmaxs.append(bbox_xmax)
        bbox_ymaxs.append(bbox_ymax)

    feature = {
        'image/height':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/depth':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
        'image/object/bbox/xmin':
        tf.train.Feature(float_list=tf.train.FloatList(value=bbox_xmins)),
        'image/object/bbox/ymin':
        tf.train.Feature(float_list=tf.train.FloatList(value=bbox_ymins)),
        'image/object/bbox/xmax':
        tf.train.Feature(float_list=tf.train.FloatList(value=bbox_xmaxs)),
        'image/object/bbox/ymax':
        tf.train.Feature(float_list=tf.train.FloatList(value=bbox_ymaxs)),
        'image/object/class/label':
        tf.train.Feature(int64_list=tf.train.Int64List(value=class_ids)),
        'image/object/class/text':
        tf.train.Feature(bytes_list=tf.train.BytesList(value=class_texts)),
        'image/encoded':
        _bytes_feature(content),
        'image/filename':
        _bytes_feature(os.path.basename(filename).encode())
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


@ray.remote
def _build_single_tf_record(chunk, path) -> None:
    print('start to build tf records for ' + path)
    with tf.io.TFRecordWriter(path) as writer:
        for anno_list in chunk:
            tf_example = _generate_tf_example(anno_list)
            writer.write(tf_example.SerializeToString())
    print('finished building tf records for ' + path)


def build_tf_records(annotations, total_shards, split) -> None:
    annotations_by_image = {}
    for annotation in annotations:
        if annotation['filename'] in annotations_by_image:
            annotations_by_image[annotation['filename']].append(annotation)
        else:
            annotations_by_image[annotation['filename']] = [annotation]
    chunks = _chunkify(list(annotations_by_image.values()), total_shards)
    futures = [
        # train_0001_of_0064.tfrecords
        _build_single_tf_record.remote(
            chunk, './tfrecords/{}_{}_of_{}.tfrecords'.format(
                split,
                str(i + 1).zfill(4),
                str(total_shards).zfill(4),
            )) for i, chunk in enumerate(chunks)
    ]
    ray.get(futures)


def _parse_one_annotation(anno, categories, dir_path) -> Dict:
    category_id = int(anno['category_id'])
    category = categories[category_id]
    class_id = category[0]
    if class_id < 0:
        print('ALERT: class is {} is invalid'.format(class_id))
    class_text = category[1]
    bbox = anno['bbox']
    filename = '{}/{}.jpg'.format(dir_path, str(anno['image_id']).rjust(12, '0'))
    annotation = {
        'filename': filename,
        'class_id': class_id,
        'class_text': class_text,
        'xmin': float(bbox[0]),
        'ymin': float(bbox[1]),
        'xmax': float(bbox[0]) + float(bbox[2]),
        'ymax': float(bbox[1]) + float(bbox[3]),
    }
    return annotation


def preprocess_coco(annot_dir: str) -> None:

    if not annot_dir == os.getcwd():
        os.chdir(annot_dir)

    print('Start to parse annotations.')

    if not os.path.exists(annot_dir + '/tfrecords'):
        os.makedirs(annot_dir + './tfrecords')

    with open(annot_dir + '/annotations/instances_train2017.json') as train_json:
        train_annos = json.load(train_json)
        train_categories = {
            category['id']: (i, category['name'])
            for i, category in enumerate(train_annos['categories'])
        }
        print(train_categories)
        train_annotations = [
            _parse_one_annotation(anno, train_categories, './train2017')
            for anno in train_annos['annotations']
        ]
        del train_annos

    with open(annot_dir + '/annotations/instances_val2017.json') as val_json:
        val_annos = json.load(val_json)
        val_categories = {
            category['id']: (i, category['name'])
            for i, category in enumerate(val_annos['categories'])
        }
        print(val_categories)
        val_annotations = [
            _parse_one_annotation(anno, val_categories, './val2017')
            for anno in val_annos['annotations']
        ]
        del val_annos

    print('Start to build TF Records.')

    build_tf_records(train_annotations, NUM_TRAIN_SHARDS, 'train')
    build_tf_records(val_annotations, NUM_VAL_SHARDS, 'val')

    print('Successfully wrote {} annotations to TF Records.'.format(
        len(train_annotations) + len(val_annotations)))


def download_coco():
    for d in ["val2017", "train2017", "test2017"]:
        if not os.path.exists(d):
            os.mkdir(d)

    os.system("gsutil -m rsync gs://images.cocodataset.org/val2017 val2017 && \
    gsutil -m rsync gs://images.cocodataset.org/train2017 train2017 && \
    gsutil -m rsync gs://images.cocodataset.org/test2017 test2017 && \
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip &&\
    unzip annotations_trainval2017.zip")


if __name__ == '__main__':
    preprocess_coco(".")
