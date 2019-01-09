import pathlib as path
import functools as func
import operator
import xml.etree.ElementTree as xml
import jsonpickle as json
from PIL import Image

class Annotation:
    def __init__(self, obj):
        self.name = obj.find('name').text
        self.difficult = bool(obj.find('difficult').text)
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
        self.filename = root.find('filename').text # image filename
        self.filePath = root.find('path').text
        self.objects = [Annotation(obj) for obj in root.findall('object')]

    def crop(self):
        # assume annotation file is in the same folder as the images
        imagePath = self.annotationFile.with_name(self.filename)
        if imagePath.is_file():
            try:
                with Image.open(imagePath) as image:
                    for obj in self.objects:
                        region = image.crop((obj.xmin, obj.ymin, obj.xmax, obj.ymax))
                        # todo: naming
                        # region.save("{}.jpg".format(obj.name))
                        print("{}.jpg".format(obj.name))
            except IOError:
                print("error occurred while processing {}".format(imagePath))
                
                
        else:
            print("{} not found!".format(imagePath))
        
annotations = [path for path in path.Path('.').glob('**/*_timestamped.xml')]
annotations = [Annotations(annotation) for annotation in annotations]
for annotation in annotations:
    annotation.crop()