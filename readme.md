# Road Sign Recognition

This is a simple road sign recognition system utilizing `PCA`, `LDA`, `MLP`, `k-NN`, etc. from `scikit-learn`.

## Data pre-processing

`main.py` contains the entrypoints of commands for counting, cropping, renaming, processing and augmenting data.

```console
usage: TPR2251 Road Sign Recognition [-h]
                                     {crop,rename,count,preprocess,augment}
                                     ...

positional arguments:
  {crop,rename,count,preprocess,augment}
    crop                crop images under /data according to annotation files
                        and save them to /cropped
    rename              copy and rename images from /raw to /data
    count               count the number of images of each label (only images
                        with _timestamp in its filename are counted)
    preprocess          preprocess all images under /cropped and copy them to
                        /preprocessed
    augment             augment all images under /cropped and save them to
                        /aug, then preprocess them and save them to
                        /preprocessed

optional arguments:
  -h, --help            show this help message and exit
```

Typical pipeline: `count -> rename -> (annotate with labelImg) -> crop -> preprocess or augment`

```console
$ python main.py count
              Morning    Noon    Night    Total
----------  ---------  ------  -------  -------
No entry           22      22        8       52
One way            44      60       84      188
Speed bump        100     140      220      460
Stop               80     104      116      300
```

```console
$ python main.py rename
Copied raw\No entry\Morning\IMG_4161-0001_timestamped.jpg to data\No entry\0001-No entry-Morning.jpg
Copied raw\No entry\Morning\IMG_4161-0002_timestamped.jpg to data\No entry\0002-No entry-Morning.jpg
...
1000 files copied
```

```console
$ python main.py crop
0001-No entry.jpg
0002-No entry.jpg
0003-No entry.jpg
...
```

```console
$ python main.py preprocess
2019-02-20 01:10:37.782 | DEBUG    | rsr:preprocessDir:190 - Preprocessing images under /cropped
2019-02-20 01:10:37.784 | DEBUG    | rsr:preprocessDir:191 - Target size: (32, 32)
2019-02-20 01:10:37.787 | DEBUG    | rsr:load:93 - Reading 0001-No entry.jpg
2019-02-20 01:10:37.799 | DEBUG    | rsr:load:93 - Reading 0002-No entry.jpg
...
2019-02-20 01:12:00.905 | DEBUG    | rsr:preprocessDir:203 - Done preprocessing
2019-02-20 01:12:00.908 | DEBUG    | rsr:pickleData:213 - Pickling data
2019-02-20 01:12:00.918 | DEBUG    | rsr:pickleData:215 - images: (486, 1024)
2019-02-20 01:12:00.920 | DEBUG    | rsr:pickleData:216 - labels: (486,)
2019-02-20 01:12:00.934 | DEBUG    | rsr:pickleData:224 - Pickled data
```

## Recognition

Most of the models defined lives in the Jupyter notebook `rsr.ipynb`. The code in the notebook will `unpickle` data (saved under `/pkl/`) processed by the command line tool (`main.py`) and pass them to those models.

1. `k-NN`
1. `PCA-kNN`
1. `LDA-kNN`
1. `PCA-LDA-kNN`
1. `PCA-SVM`
1. `PCA-LDA-SVM`
1. `MLP`
1. `PCA-MLP`
1. `LDA-MLP`
1. `PCA-LDA-MLP`

> Almost every of the code written for experimenting and testing are saved inside `notebook.py`

## Data

Raw data are not included due to their size. Here is how data are organized:

### `raw`

Raw data, `main.py` will only count images with `_timestamped` in its filename

```
.
└── raw
    ├── <class>
    │   ├── Morning
    │   │   └── *_timestamped.jpg
    │   ├── Noon
    │   │   └── ...
    │   └── Night
    │       └── ...
    ├── <class>
    │   └── ...
    └── <class>
        └── ...
```

### `data`

> Generated from `raw` with the `rename` command

Renamed raw data. Also contains annotations files generated using `labelImg`.

```
.
└── data
    ├── <class>
    │   ├── <id>-<class>-<time>.jpg
    │   ├── <id>-<class>-<time>.xml
    │   └── ...
    ├── <class>
    │   └── ...
    └── <class>
        └── ...
```

### `cropped`

> Generated from `data` with the `crop` command

Cropped images from `data` according to their annotation files.

```
.
└── cropped
    ├── <class>
    │   ├── <id>-<class>-<time>[-difficult].jpg
    │   └── ...
    └── <class>
        └── ...
```

### `aug`

> Generated from `data` with the `augment` command

Augmented images from  `data`.

```
.
└── aug
    ├── <class>
    │   ├── <id>-<class>-<time>[-difficult].jpg
    │   └── ...
    └── <class>
        └── ...
```

### `preprocessed`

> Generated from `data` using `preprocess` command

Grayscaled, resized, equalized images from `data`

```
.
└── preprocessed
    ├── <id>-<class>.jpg
    └── ...
```

### `pkl`

> Generated automatically by `preprocess` and `augment` command

Serialized preprocessed images

```
.
└── pkl
    ├── images_aug.pkl
    ├── images.pkl
    ├── labels_aug.pkl
    └── labels.pkl
```

All of the directory variables are defined in `rsr.py`:
```python
RAW_DATA_DIRECTORY = "raw" # images directly from the camera
DATA_DIRECTORY = "data" # renamed images, with annotations
CROPPED_DIRECTORY = "cropped" # images cropped according to annotations
PREPROCESSED_DIRECTORY = "preprocessed" # images that are proprocessed (equalize, grayscale, etc)
AUG_DIRECTORY = "aug" # augmented images
PICKLE_DIRECTORY = "pkl" # store serialized dataset
LABELS = ["No entry", "One way", "Speed bump", "Stop"]
```

## Others

This project is made for my `TPR2251 Pattern Recognition` subject.