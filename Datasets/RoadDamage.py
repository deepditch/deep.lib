import Datasets.ModelData as md
import Datasets.ImageData as ImageData
import Datasets.ClassifierData as ClassifierData
import itertools
import torchvision
from torchvision import datasets, models, transforms
from pathlib import Path
from Transforms.ImageTransforms import *
from xml.dom import minidom
import pandas as pd
from collections import namedtuple, OrderedDict
import pickle
import os 

def RoadDamageDataset(data_path, imsize=224, batch_size=8, partitions={'train': .9, 'valid': .1}):
    DATA_PATH = Path(data_path)
    MULTICLASS_CSV_PATH = DATA_PATH/'mc.csv'
    MULTIBB_CSV_PATH = DATA_PATH/'bb.csv'
    files, mcs = ImageData.parse_csv_data(MULTICLASS_CSV_PATH)
    files, mbbs = ImageData.parse_csv_data(MULTIBB_CSV_PATH)
    mcs = [mc.split(' ') for mc in mcs]
    classes = ["bg"] + sorted(list(set(itertools.chain.from_iterable(mcs))))
    label2idx = {v:k for k,v in enumerate(classes)}
    mcs = [[label2idx[c] for c in mc] for mc in mcs]
    mbbs = [corners_to_center([int(x) for x in mbb.split(' ')]).tolist() for mbb in mbbs]

    max_len = max([len(mc) for mc in mcs])

    for mc, mb in zip(mcs, mbbs):
        mc += ([-1] * (max_len - len(mc)))
        mb += ([0] * (max_len * 4 - len(mb)))
        
        mc = np.array(mc)
        mb = np.array(mb)

    num_classes = 8

    files = [DATA_PATH/file for file in files]
    labels = [md.StructuredLabel([(cat, md.LabelType.CATEGORY, "CAT"), (bb, md.LabelType.BOUNDING_BOX, "BB")]) for bb, cat in zip(mbbs, mcs)]

    train_tfms = TransformList([
        RandomScale(imsize, 1.17),
        RandomCrop(imsize),
        RandomHorizontalFlip(),
        RandomLighting(.05, .05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_tfms = TransformList([
        Scale(imsize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    denorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    
    if not os.path.exists(DATA_PATH/'train_val_split.pickle'):
        i_dict = md.make_partition_indices(len(labels), partitions)
        with open(DATA_PATH/'train_val_split.pickle', 'wb') as handle:
            pickle.dump(i_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    else:
        with open(DATA_PATH/'train_val_split.pickle', 'rb') as handle:
            i_dict = pickle.load(handle)
    
    idx, test_files = ImageData.parse_csv_data(DATA_PATH/'test_data.csv')
    test_files = [DATA_PATH/file.replace("\\", "/") for file in test_files]

    datasets = {
        'train': ImageData.ImageDataset(util.mask(files, i_dict['train']), util.mask(labels, i_dict['train']), train_tfms),
        'valid': ImageData.ImageDataset(util.mask(files, i_dict['valid']), util.mask(labels, i_dict['valid']), val_tfms),
        'test': ImageData.ImageDataset(test_files, [0] * len(test_files), val_tfms)
    }  

    return md.ModelData(datasets, batch_size), classes, train_tfms, val_tfms, denorm


def ParseDataFiles(files, csv):
    data = []
    for file in files:
        doc = minidom.parse(file)  
        anno = doc.getElementsByTagName('annotation')[0]
        folder = anno.getElementsByTagName('folder')[0].firstChild.nodeValue
        filename = folder + "/JPEGImages/" + anno.getElementsByTagName('filename')[0].firstChild.nodeValue
        size = anno.getElementsByTagName('size')[0]
        width = size.getElementsByTagName('width')[0].firstChild.nodeValue
        height = size.getElementsByTagName('height')[0].firstChild.nodeValue
        objects = anno.getElementsByTagName('object')
        for obj in objects:
            cls = obj.getElementsByTagName('name')[0].firstChild.nodeValue
            if cls == "D30": continue
            d = {"filename": filename, "width": width, "height": height, "class": cls}
            data.append(d)

    df = pd.DataFrame(data, columns=['filename', 'width', 'height', 'class'])
    df.to_csv(csv, index=False)


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def ParseDataCSV(path, csv):
    examples = pd.read_csv(path/csv)
    images = split(examples, 'filename')
    mc = [[row['class'] for index, row in img.object.iterrows()] for img in images]
    images = [path/img.filename for img in images]
    return images, mc


def RoadDamageClassifierData(data_path, imsize=224, batch_size=8, partitions={'train': .85, 'valid': .15}):  
    DATA_PATH = Path(data_path)
  
    if partitions != None:
        # Let's create a new training and validation set
        
        # These are the folders in the original dataset
        govs =  ["Adachi", "Chiba", "Ichihara", "Muroran", "Nagakute", "Numazu", "Sumida"]   
        
        # We aggregate each file from the original dataset
        original_files = []   
        for gov in govs:
            original_files.extend([os.path.join(DATA_PATH, gov, 'Annotations', file) for file in os.listdir(os.path.join(DATA_PATH, gov, 'Annotations'))])

        # We then partition the original dataset according to the passed arguments
        i_dict = md.make_partition_indices(len(original_files), partitions)
        train_files = util.mask(original_files, i_dict['train'])
        valid_files = util.mask(original_files, i_dict['valid'])

        # Then, let's add additional images (verified from the website) into the training set
        new_folders = [name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name)) and name not in govs]

        for folder in new_folders:
            train_files.extend([os.path.join(DATA_PATH, folder, 'Annotations', file) for file in os.listdir(os.path.join(DATA_PATH, folder, 'Annotations'))])

        # Then we can parse the annotation files we have aggregated into a simpler format, then save our data into a .csv
        ParseDataFiles(train_files, DATA_PATH/"train_data.csv")
        ParseDataFiles(valid_files, DATA_PATH/"valid_data.csv")


    train_images, train_labels = ParseDataCSV(DATA_PATH, "train_data.csv")
    valid_images, valid_labels = ParseDataCSV(DATA_PATH, "valid_data.csv")

    train_labels, _ = ClassifierData.make_n_hot_labels(train_labels)
    valid_labels, _ = ClassifierData.make_n_hot_labels(valid_labels)


    train_tfms = TransformList([
        RandomScale(imsize, 1.17),
        RandomCrop(imsize),
        RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_tfms = TransformList([
        Scale(imsize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    datasets = {
        'train': ImageData.ImageDataset(train_images, train_labels, train_tfms),
        'valid': ImageData.ImageDataset(valid_images, valid_labels, val_tfms)
    }  

    return md.ModelData(datasets, batch_size)