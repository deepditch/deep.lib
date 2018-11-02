import Datasets.ModelData as md
import Datasets.ImageData as ImageData
import itertools
import torchvision
from torchvision import datasets, models, transforms
from pathlib import Path
from Transforms.ImageTransforms import *

def RoadDamageDataset(data_path, imsize=224, batch_size=8, partitions={'train': .85, 'valid': .15}):
    DATA_PATH = Path('C:/fastai/courses/dl2/data/road_damage_dataset')
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
        RandomScale(imsize, 1.1),
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

    denorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )

    i_dict = md.make_partition_indices(len(labels), partitions)

    idx, test_files = ImageData.parse_csv_data(DATA_PATH/'test_data.csv')
    test_files = [DATA_PATH/file for file in test_files]

    datasets = {
        'train': ImageData.ImageDataset(util.mask(files, i_dict['train']), util.mask(labels, i_dict['train']), train_tfms),
        'valid': ImageData.ImageDataset(util.mask(files, i_dict['valid']), util.mask(labels, i_dict['valid']), val_tfms),
        'test': ImageData.ImageDataset(test_files, [0] * len(test_files), val_tfms)
    }  

    return md.ModelData(datasets, batch_size), classes, train_tfms, val_tfms, denorm