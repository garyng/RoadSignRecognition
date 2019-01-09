import pathlib as path
import functools as func
import operator
import itertools as iter
from shutil import copy2

imagesGlob = ['**/*_timestamped.jpg', '**/*_timestamped.JPG']
images = func.reduce(operator.add, [[path for path in path.Path(
    '.').glob(glob)] for glob in imagesGlob], [])
labelled = sorted([{
    'label': image.parent.parent.name,
    'time': image.parent.name,
    'path': image
} for image in images], key=lambda label: label['label'])
grouped = iter.groupby(labelled, key=lambda label: label['label'])
for label, images in grouped:
    count = 1
    for image in images:
        filename = f"{count:04d}-{label}-{image['time']}.jpg"
        directory = path.Path('./data') / label
        directory.mkdir(parents=True, exist_ok=True)
        copy2(image['path'], directory / filename)
        count += 1
