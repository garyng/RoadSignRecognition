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
                        
                        directory = path.Path('./cropped') / obj.name
                        directory.mkdir(parents=True, exist_ok=True)
                        target = directory / filename
                        
                        region.save(target)
                            
            except IOError:
                print("error occurred while processing {}".format(imagePath))

        else:
            print("{} not found!".format(imagePath))


def crop(args):
    counts = {} # to keep track of the indexes for each label
    annotations = [path for path in path.Path('.').glob('./data/**/*.xml')]
    annotations = [Annotations(annotation) for annotation in annotations]
    for annotation in annotations:
        annotation.crop(counts)

def getGroupedImages():
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
    grouped = getGroupedImages()
    total = 0
    for label, images in grouped:
        count = 1
        for image in images:
            filename = f"{count:04d}-{label}-{image['time']}.jpg"
            directory = path.Path('./data') / label
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / filename
            copy2(image['path'], target)
            print(f"Copied {image['path']} to {target}")
            count += 1
            total += 1

    print(f'{total} files copied')

def count(args):
    grouped = getGroupedImages()
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

parser = argparse.ArgumentParser(prog="TPR2251 Road Sign Recognition")
subparsers = parser.add_subparsers()

# crop
cropParser = subparsers.add_parser(
    'crop', help="crop raw images under /data according to annotation files and save them to /cropped")
cropParser.set_defaults(func=crop)

# rename
renameParser = subparsers.add_parser(
    'rename', help="copy and rename images from /raw to /data")
renameParser.set_defaults(func=rename)

# count
countParser = subparsers.add_parser('count', help="count the number of raw images of each label")
countParser.set_defaults(func=count)

args = parser.parse_args()
args.func(args)
