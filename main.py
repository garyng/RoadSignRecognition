from rsr import *

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

# augment
augmentParser = subparsers.add_parser('augment', help=f"augment all images under /{CROPPED_DIRECTORY} and save them to /{AUG_DIRECTORY}, then preprocess them and save them to /{PREPROCESSED_DIRECTORY}")
augmentParser.add_argument('--count', type=int, default=300)
augmentParser.add_argument('--width', type=int, default=32, help="Target width")
augmentParser.add_argument('--height', type=int, default=32, help="Target height")
augmentParser.set_defaults(func=augment)

args = parser.parse_args()
args.func(args)
