import argparse
import pathlib as path
import functools as func
import operator
import xml.etree.ElementTree as xml
import jsonpickle as json
from PIL import Image
import pathlib as path
import itertools as iter
from shutil import copy2
from tabulate import tabulate
import skimage as skimg
import matplotlib.pyplot as plt
from functional import seq
from random import sample, seed
import numpy as np
from loguru import logger

RAW_DATA_DIRECTORY = "raw" # images directly from the camera
DATA_DIRECTORY = "data" # renamed images, with annotations
CROPPED_DIRECTORY = "cropped" # images cropped according to annotations
PREPROCESSED_DIRECTORY = "preprocessed" # images that are proprocessed (equalize, grayscale, etc)
LABELS = ["No entry", "One way", "Speed bump", "Stop"]


class Annotation:
    def __init__(self, obj):
        self.name = obj.find('name').text
        self.difficult = bool(int(obj.find('difficult').text))
        boundingBox = obj.find('bndbox')
        self.xmin = int(round(float(boundingBox.find('xmin').text)))
        self.ymin = int(round(float(boundingBox.find('ymin').text)))
        self.xmax = int(round(float(boundingBox.find('xmax').text)))
        self.ymax = int(round(float(boundingBox.find('ymax').text)))

class Annotations:
    def __init__(self, annotationFile):
        self.annotationFile = annotationFile
        self.parse()

    def parse(self):
        tree = xml.parse(self.annotationFile)
        root = tree.getroot()
        # self.folder = root.find('folder').text
        self.filename = root.find('filename').text  # image filename
        self.filePath = root.find('path').text
        self.objects = [Annotation(obj) for obj in root.findall('object')]

    def crop(self, indexSource):
        # assume annotation file is in the same folder as the images
        imagePath = self.annotationFile.with_name(self.filename)
        if imagePath.is_file():
            try:
                with Image.open(imagePath) as image:
                    for obj in self.objects:
                        region = image.crop(
                            (obj.xmin, obj.ymin, obj.xmax, obj.ymax))

                        if obj.name not in indexSource:
                            indexSource[obj.name] = 1
                        else:
                            indexSource[obj.name] += 1

                        filename =f'{indexSource[obj.name]:04d}-{obj.name}{"-difficult" if obj.difficult else ""}.jpg'
                        print(filename)
                        
                        directory = path.Path(f'./{CROPPED_DIRECTORY}') / obj.name
                        directory.mkdir(parents=True, exist_ok=True)
                        target = directory / filename
                        
                        region.save(target)
                            
            except IOError:
                print("error occurred while processing {}".format(imagePath))

        else:
            print("{} not found!".format(imagePath))

class Data:
    labels = LABELS

    def __init__(self, imagePath):
        self.path = imagePath
        self.load()
    
    def load(self):
        """Load the image from disk into memory and determines its label from the directory name"""
        logger.debug(f"Reading {self.path.name}")
        self.label = int(Data.fromLabel(self.path.parent.name))
        self.image = skimg.data.imread(self.path)
    
    def preprocess(self, size = (32, 32)):
        self.grayscale()
        self.equalize()
        self.resize()

    def save(self):
        directory = path.Path(f'./{PREPROCESSED_DIRECTORY}')
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / self.path.name
        skimg.io.imsave(target, self.image)
        logger.debug(f"Saved {target}")

    def grayscale(self):
        self.image = skimg.img_as_ubyte(skimg.color.rgb2gray(self.image))

    def equalize(self):
        self.image = skimg.img_as_ubyte(skimg.exposure.equalize_adapthist(self.image))

    def resize(self, size = (32, 32)):
        self.image = skimg.transform.resize(self.image, size, mode='constant', anti_aliasing=True)

    @staticmethod
    def fromLabel(name):
        """Convert label name to index"""
        return Data.labels.index(name)
    
    @staticmethod
    def fromIndex(index):
        """Convert index to label name"""
        return Data.labels[index]


def crop(args):
    counts = {} # to keep track of the indexes for each label
    annotations = [path for path in path.Path('.').glob(f'./{DATA_DIRECTORY}/**/*.xml')]
    annotations = [Annotations(annotation) for annotation in annotations]
    for annotation in annotations:
        annotation.crop(counts)

def getGrouppedRawImages():
    """Returns image paths grouped by their label"""
    imagesGlob = ['**/*_timestamped.jpg', '**/*_timestamped.JPG']
    images = func.reduce(operator.add, [[path for path in path.Path(
        '.').glob(glob)] for glob in imagesGlob], [])
    labelled = sorted([{
        'label': image.parent.parent.name,
        'time': image.parent.name,
        'path': image
    } for image in images], key=lambda label: label['label'])
    return iter.groupby(labelled, key=lambda label: label['label'])

def rename(args):
    grouped = getGrouppedRawImages()
    total = 0
    for label, images in grouped:
        count = 1
        for image in images:
            filename = f"{count:04d}-{label}-{image['time']}.jpg"
            directory = path.Path(f'./{DATA_DIRECTORY}') / label
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / filename
            copy2(image['path'], target)
            print(f"Copied {image['path']} to {target}")
            count += 1
            total += 1

    print(f'{total} files copied')

def count(args):
    grouped = getGrouppedRawImages()
    stats = {}
    for label, images in grouped:
        times = iter.groupby(sorted(images, key=lambda image: image['time']), key=lambda image: image['time'])
        statsPerTime = {}
        subTotal = 0
        for time, images in times:
            count = len(list(images))
            statsPerTime[time] = count
            subTotal += count

        statsPerTime['Total'] = subTotal
        stats[label] = statsPerTime

    # [['label', morning, noon, night, total], ...]
    headers = ['Morning', 'Noon', 'Night', 'Total']
    transformed = [[stat] + [stats[stat][time] for time in headers] for stat in stats]
    print(tabulate(transformed, headers=headers))

def preprocess(args):
    size = (args.height, args.width)
    logger.debug(f'Preprocessing images under /{CROPPED_DIRECTORY}')
    logger.debug(f'Target size: {size}')

    dataset = [Data(path) for path in path.Path('.').glob(f'./{CROPPED_DIRECTORY}/**/*.jpg')]

    groups = seq(dataset).group_by(lambda data: data.label)
    for group in groups:
        logger.info(f'{Data.fromIndex(group[0])}: {len(group[1])}')
    logger.info(f'Total: {len(dataset)}')

    for data in dataset:
        data.preprocess(size=size)
        data.save()
    logger.debug("Done preprocessing")


parser = argparse.ArgumentParser(prog="TPR2251 Road Sign Recognition")
subparsers = parser.add_subparsers()

# crop
cropParser = subparsers.add_parser(
    'crop', help=f"crop images under /{DATA_DIRECTORY} according to annotation files and save them to /{CROPPED_DIRECTORY}")
cropParser.set_defaults(func=crop)

# rename
renameParser = subparsers.add_parser(
    'rename', help=f"copy and rename images from /{RAW_DATA_DIRECTORY} to /{DATA_DIRECTORY}")
renameParser.set_defaults(func=rename)

# count
countParser = subparsers.add_parser('count', help="count the number of images of each label (only images with _timestamp in its filename are counted)")
countParser.set_defaults(func=count)

# preprocess
preprocessParser = subparsers.add_parser('preprocess', help=f"preprocess all images under /{CROPPED_DIRECTORY} and copy them to /{PREPROCESSED_DIRECTORY}")
preprocessParser.add_argument('--width', type=int, default=32)
preprocessParser.add_argument('--height', type=int, default=32)
preprocessParser.set_defaults(func=preprocess)

args = parser.parse_args(['preprocess'])
args.func(args)
