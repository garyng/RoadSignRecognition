#%% [markdown]
# # Road Sign Recognition system

#%%
import pathlib as path
import skimage as skimg
import matplotlib.pyplot as plt
from functional import seq
from random import sample, seed

#%%
class Data:
    labels = ["No entry", "One way", "Speed bump", "Stop"]

    def __init__(self, path):
        self.path = path
        self.load(path)

    def load(self, path):
        self.label = int(Data.fromLabel(path.parent.name))
        self.image = skimg.data.imread(path)
    
    def resize(self, size = (32, 32)):
        self.image = skimg.transform.resize(self.image, size, mode='constant')

    @staticmethod
    def fromLabel(name):
        """Convert label name to index"""
        return Data.labels.index(name)
    
    @staticmethod
    def fromIndex(index):
        """Convert index to label name"""
        return Data.labels[index]


#%% [markdown]
# # Loading dataset
# Load the data from the `/cropped/` directory, convert the directory name to integer and used them as labels

#%%
paths = path.Path('.').glob('./cropped/**/*.jpg')
dataset = []
for path in paths:
    dataset.append(Data(path))
groups = seq(dataset).group_by(lambda data: data.label)

#%% [markdown]
# Dataset distribution

#%%
for group in groups:
    print(f'{Data.fromIndex(group[0])}: {len(group[1])}')

#%% [markdown]
# # Visualizing dataset
# Displaying 10 random samples for each label
# 
# _Here, the random seed is set to 0_

#%%
seed(0)
def showDataset(dataset, title, cols = 10, figsize = (15, 15)):
    rows = len(dataset) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    for axe, image in zip(axes, seq(dataset).select(lambda data: data.image)):
        axe.axis('off')
        axe.imshow(image)
    
    fig.suptitle(title, y=0.6)
    plt.show()
    return

def sampleDataset(count=10):
    for group in groups:
        showDataset(sample(group[1], count), Data.fromIndex(group[0]))
    return

sampleDataset()


#%% [markdown]
# Image resizing
#

#%%
def minMaxAverage(title, items):
    print(f'Minimum {title}: {seq(items).min()}')
    print(f'Maximum {title}: {seq(items).max()}')
    print(f'Average {title}: {seq(items).average()}')
    
minMaxAverage("width", seq(dataset).select(lambda data: data.image.shape[0]))
minMaxAverage("height", seq(dataset).select(lambda data: data.image.shape[1]))

#%%
for data in dataset:
    data.resize()

#%%
minMaxAverage("width", seq(dataset).select(lambda data: data.image.shape[0]))
minMaxAverage("height", seq(dataset).select(lambda data: data.image.shape[1]))

sampleDataset()

# for (ax, (index, image)) in zip(axs, enumerate(labels)):
#     ax.set_title("hi")

# A = np.random.rand(5, 5)

# fig, axs = plt.subplots(1, 3, figsize=(10, 3))
# for ax, interp in zip(axs, ['nearest', 'bilinear', 'bicubic']):
#     ax.imshow(A, interpolation=interp)
#     ax.set_title(interp.capitalize())
#     ax.grid(True)

# plt.show()


#%%
# import numpy as np
# #First create some toy data:
# x = np.linspace(0, 2*np.pi, 400)
# y = np.sin(x**2)
# #Creates just a figure and only one subplot
# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.set_title('Simple plot')

# #Creates two subplots and unpacks the output array immediately
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(x, y)
# ax1.set_title('Sharing Y axis')
# ax2.scatter(x, y)

# #Creates four polar axes, and accesses them through the returned array
# fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))
# axes[0, 0].plot(x, y)
# axes[1, 1].scatter(x, y)

# #Share a X axis with each column of subplots
# plt.subplots(2, 2, sharex='col')
# #Share a Y axis with each row of subplots
# plt.subplots(2, 2, sharey='row')

# #Share both X and Y axes with all subplots
# plt.subplots(2, 2, sharex='all', sharey='all')
# #Note that this is the same as
# plt.subplots(2, 2, sharex=True, sharey=True)
# #Creates figure number 10 with a single subplot
# #and clears it if it already exists.
# fig, ax=plt.subplots(num=10, clear=True)
# #%%
