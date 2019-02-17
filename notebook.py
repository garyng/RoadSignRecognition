#%% [markdown]
# # Road Sign Recognition system

#%%
import pathlib as path
import skimage as skimg
import matplotlib.pyplot as plt
from functional import seq
from random import sample, seed
import numpy as np
from rsr import Data
from loguru import logger
import Augmentor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import svm as SVM
from mpl_toolkits import mplot3d
from sklearn.metrics import classification_report

logger.remove()

#%% [markdown]
# # Loading dataset
# Load the data from the `/cropped/` directory, convert the directory name to integer and used them as labels

#%%
def loadDataset():
    paths = path.Path('.').glob('./cropped/**/*.jpg')
    dataset = []
    for p in paths:
        dataset.append(Data(p))
    return dataset

def groupDataset(dataset):
    groups = seq(dataset).group_by(lambda data: data.label)
    return groups

dataset = loadDataset()
groups = groupDataset(dataset)

#%% [markdown]
# # Dataset distribution

#%%
def printStat(dataset, groups):
    for group in groups:
        print(f'{Data.fromIndex(group[0])}: {len(group[1])}')
    print(f'Total: {len(dataset)}')

    plt.bar(seq(groups).select(lambda group: Data.fromIndex(group[0])).to_list(), seq(groups).select(lambda group: len(group[1])).to_list())
    plt.show()

printStat(dataset, groups)
#%% [markdown]
# # Visualizing dataset
# Displaying 10 random samples for each label
# 
# _Here, the random seed is set to 0_

#%%
seed(0)

def showImages(images, title, cols = 10, figsize = (15, 15), y = 0.6):
    rows = len(images) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    cmap = None
    for axe, image in zip(axes.ravel(), images):
        axe.axis('off')
        if len(image.shape) < 3 or image.shape[-1] < 3:
            cmap = "gray"
        axe.imshow(image, cmap = cmap)
    
    fig.suptitle(title, y=y)
    plt.show()
    return

def showDataset(dataset, title, cols = 10, figsize = (15, 15), y = 0.6):
    showImages(seq(dataset).select(lambda data: data.image).to_list(), title, cols, figsize, y)
    
def sampleDataset(dataset, count=10):
    groups = groupDataset(dataset)
    for group in groups:
        showDataset(sample(group[1], count), Data.fromIndex(group[0]), cols=count)
    return

#%%
sampleDataset(dataset)

#%% [markdown]
# # Image preprocessing

#%%
def minMaxAverage(title, items):
    print(f'Minimum {title}: {seq(items).min()}')
    print(f'Maximum {title}: {seq(items).max()}')
    print(f'Average {title}: {seq(items).average()}')

#%% [markdown]
# Average width

#%%
minMaxAverage("width", seq(dataset).select(lambda data: data.image.shape[0]))

#%% [markdown]
# Average height

#%%
minMaxAverage("height", seq(dataset).select(lambda data: data.image.shape[1]))

#%% [markdown]
# # Validation set
# Here, we only sample 100 images for testing out the preprocessing steps

#%%
# validation set
validationSet = sample(dataset, 100)
showDataset(validationSet, "Validation set", 5, y = 0.9)

#%% [markdown]
# Grayscale

#%% 
for data in validationSet:
    data.grayscale()

showDataset(validationSet, "Grayscale", 5, y = 0.9)

#%% [markdown]
# Histogram equalization

#%% 
for data in validationSet:
    data.equalize()

showDataset(validationSet, "Histogram equalization", 5, y = 0.9)

#%% [markdown]
# Resize

#%%
for data in validationSet:
    data.resize()

showDataset(validationSet, "Resize", 5, y = 0.9)

#%% [markdown]
# Final width

#%%
minMaxAverage("width", seq(validationSet).select(lambda data: data.image.shape[0]))

#%% [markdown]
# Final height

#%%
minMaxAverage("height", seq(validationSet).select(lambda data: data.image.shape[1]))


#%% [markdown]
# Saving preprocessed images to the directory /preprocessed

#%% [markdown]
# Pickling preprocessed data

#%%
import pickle

def pickleData():
    dataset = loadDataset()
    for data in dataset:
        data.preprocess()

    images, labels = getDataLabelsFromDataset(dataset)

    print(images.shape)
    print(labels.shape)

    pickle.dump(images, open('images.pkl', 'wb'))
    pickle.dump(labels, open('labels.pkl', 'wb'))

#%%
pickleData()

# todo: change to "/pkl/" directory
#%%
def unpickleData():
    images = pickle.load(open("images.pkl", "rb"))
    labels = pickle.load(open("labels.pkl", "rb"))
    return (images, labels)


#%% [markdown]
# # MLP

#%%

def getDataLabelsFromDataset(dataset):
    data = np.array(seq(dataset).select(lambda data: data.image).to_list()).reshape(len(dataset), -1)
    labels = np.array(seq(dataset).select(lambda data: data.label).to_list()).reshape(-1)
    return (data, labels)

# todo: change to use pickled data
# def trainMlp(dataset):
#     # data, labels = getDataLabelsFromDataset(dataset)
    
#     # data, labels = augmentData(data.reshape(-1, 32, 32), labels, batch_size=5000)
#     # data = data.reshape(len(data), -1)

#     trainingData, testingData, trainingLabels, testingLabels = train_test_split(data, labels, random_state=0)
#     print(f'trainingData: {trainingData.shape}, trainingLabels: {trainingLabels.shape}')
#     print(f'testingData: {testingData.shape}, testingLabels: {testingLabels.shape}')

#     minMaxAverage("training", trainingData.ravel())
#     minMaxAverage("testing", testingData.ravel())

#     scaler = StandardScaler()
#     trainingData = scaler.fit_transform(trainingData)
#     testingData = scaler.fit_transform(testingData)
#     print("---")
#     minMaxAverage("training", trainingData.ravel())
#     minMaxAverage("testing", testingData.ravel())

#     mlp = MLPClassifier(random_state=0, verbose=True)
#     mlp.fit(trainingData, trainingLabels)
#     print(mlp)
#     print(f'training accuracy: {mlp.score(trainingData, trainingLabels)}')
#     print(f'testing accuracy: {mlp.score(testingData, testingLabels)}')

#     predictedLabels = mlp.predict(testingData)
#     print(f'confusion matrix: \n {confusion_matrix(testingLabels, predictedLabels)}')

# todo: load directly from local files
# def trainWithAllData():
#     dataset = loadDataset()
#     for data in dataset:
#         data.preprocess()
#     trainMlp(dataset)


#%%
# trainWithAllData()
# trainMlp(validationSet)


#%% [markdown]
# Data augmentation

#%%
dataset = loadDataset()[:10]
for data in dataset:
    data.preprocess()
    
#%%
data = seq(dataset).select(lambda d: d.image).to_list()
labels = seq(dataset).select(lambda d: d.label).to_list()

#%%
# todo: only accept images of [n, d]
def augmentData(data, labels, batch_size = 128, total = 512, width = 32, height = 32, samples = 50):
    # reshape 1d data
    data = list(data.reshape(-1, width, height))
    showImages(sample(data, samples), "Original")
    p = Augmentor.Pipeline()

    # p.random_distortion(probability=1, grid_width=8, grid_height=8, magnitude=1)
    p.skew(probability=0.5)
    p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
    p.rotate_random_90(probability=0.1)
    p.flip_random(probability=0.5)
    g = p.keras_generator_from_array(data, labels, batch_size)
    augmentedImages, augmentedLabels = next(g)
    count = len(augmentedLabels)
    while count < total:
        images, labels = next(g)
        augmentedImages = np.concatenate((augmentedImages, images))
        augmentedLabels = np.concatenate((augmentedLabels, labels))
        count += len(labels)
    
    showImages(sample(list(augmentedImages.reshape(-1, width, height)), samples), "Augmented")

    return (augmentedImages.reshape(count, -1), augmentedLabels)

#%%
augmentData(*unpickleData()[:10])


#%% [markdown]
# # Loading preprocessed data from disk

#%%
images, labels = unpickleData()
print(images.shape)
print(labels.shape)

#%% [markdown]
# # PCA

#%%
# find the mean image
plt.imshow(images.mean(0).reshape(32, 32), cmap="gray")

#%%


def fitPca(data, labels):
    # data, labels = augmentData(data, labels, total = 1024)

    trainingData, testingData, trainingLabels, testingLabels = train_test_split(data, labels)
    print(f'trainingData: {trainingData.shape}, trainingLabels: {trainingLabels.shape}')
    print(f'testingData: {testingData.shape}, testingLabels: {testingLabels.shape}')
    scaler = StandardScaler()

    trainingData = scaler.fit_transform(trainingData)
    testingData = scaler.fit_transform(testingData)

    pca = PCA(n_components=50, svd_solver='randomized', whiten=True)
    pca.fit(data)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    print(f'pca components: {pca.components_.shape}')
    showImages(np.array(sample(list(trainingData), 10)).reshape(-1, 32, 32), "Reconstructed images")
    plt.show()

    trainingData = pca.transform(trainingData)
    testingData = pca.transform(testingData)

    plt.scatter(trainingData[:, 0], trainingData[:, 1], c = trainingLabels, cmap="Accent")
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.show()

    return (trainingData, testingData, trainingLabels, testingLabels)

fitPca(*unpickleData())
#%%

def crossValidateKnn(data, labels):
    # https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/
    scores = []
    neighbours = seq(range(1, 50)).filter(lambda i: i % 2 == 1).to_list()
    for k in neighbours:
        knn = KNeighborsClassifier(n_neighbors=k)
        # 10-fold
        scores.append(cross_val_score(knn, data, labels, cv=10, scoring='accuracy').mean())

    mse = [1 - x for x in scores] # miss-classification error
    plt.xlabel('k')
    plt.ylabel('error rate')
    plt.plot(neighbours, mse)

    optimalK = neighbours[mse.index(min(mse))]
    print(f'optimal k: {optimalK}')

    return optimalK


def trainKnn(data, labels):
    trainingData, testingData, trainingLabels, testingLabels = fitPca(data, labels)
    k = crossValidateKnn(trainingData, trainingLabels)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(trainingData, trainingLabels)
    print(f'training accuracy: {knn.score(trainingData, trainingLabels)}')
    print(f'testing accuracy: {knn.score(testingData, testingLabels)}')

    predictedLabels = knn.predict(testingData)
    print(f'confusion matrix: \n {confusion_matrix(testingLabels, predictedLabels)}')

trainKnn(*unpickleData())


#%%
augmentData(*unpickleData())

#%%

# https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
def tsneVisualize(data, labels):
    # data, labels = augmentData(data, labels, total = 4096)
    data, _, labels, _ = fitPca(data, labels)

    tsne = TSNE(verbose=1, random_state=0, n_iter=2000)
    data = tsne.fit_transform(data)

    plt.scatter(data[:, 0], data[:, 1], c = labels, cmap="Accent", s=16)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

tsneVisualize(*unpickleData())
 
#%% 
dataset = loadDataset()
for data in dataset:
    data.preprocess()

#%% 
def augmentDataset(dataset):
    groups = groupDataset(dataset)
    max = seq(groups).select(lambda group : len(group[1])).max()
    images = []
    labels = []
    for group in groups:
        count = max - len(group[1])
        print(f'augmenting {count} images for {Data.fromIndex(group[0])}')
        rawImages = seq(group[1]).select(lambda item: item.image).to_list()
        rawImages = np.array(rawImages)
        augImg, augLbl = augmentData(rawImages, seq(group[1]).select(lambda item: item.label).to_list(), samples=10, total=count)
        images.extend(augImg)
        labels.extend(augLbl)
   
    return (images, labels)
    

#%%
trainKnn(images, labels)


#%%
print(images.shape)
print(labels.shape)

#%%
def trainMlp(data, labels):
    trainingData, testingData, trainingLabels, testingLabels = train_test_split(data, labels)
    print(f'trainingData: {trainingData.shape}, trainingLabels: {trainingLabels.shape}')
    print(f'testingData: {testingData.shape}, testingLabels: {testingLabels.shape}')

    minMaxAverage("training", trainingData.ravel())
    minMaxAverage("testing", testingData.ravel())

    scaler = StandardScaler()
    trainingData = scaler.fit_transform(trainingData)
    testingData = scaler.fit_transform(testingData)
    print("---")
    minMaxAverage("training", trainingData.ravel())
    minMaxAverage("testing", testingData.ravel())

    mlp = MLPClassifier(random_state=0, verbose=True)
    mlp.fit(trainingData, trainingLabels)
    print(mlp)
    print(f'training accuracy: {mlp.score(trainingData, trainingLabels)}')
    print(f'testing accuracy: {mlp.score(testingData, testingLabels)}')

    predictedLabels = mlp.predict(testingData)
    print(f'confusion matrix: \n {confusion_matrix(testingLabels, predictedLabels)}')

#%%
images, labels = augmentDataset(dataset)
images = np.array(images)
labels = np.array(labels)

#%%
trainMlp(images, labels)

#%%
trainKnn(images, labels)

#%%
tsneVisualize(images, labels)

#%%

def fitLda2(data, labels):
    print(f'data: {data.shape}, labels: {labels.shape}')

    data = StandardScaler().fit_transform(data)

    lda = LDA(n_components=3)
    data = lda.fit_transform(data, labels)

    scatter3d(data[:, 0], data[:, 1],data[:, 2], labels)

    return (data, labels)

def fitLda3(trainingData, testingData, trainingLabels, testingLabels):
    print(f'trainingData: {trainingData.shape}, trainingLabels: {trainingLabels.shape}')
    print(f'testingData: {testingData.shape}, testingLabels: {testingLabels.shape}')

    scaler = StandardScaler()
    trainingData = scaler.fit_transform(trainingData)
    testingData = scaler.fit_transform(testingData)

    lda = LDA(n_components=3)
    lda.fit(trainingData, trainingLabels)

    scatter3d(trainingData[:, 0], trainingData[:, 1],trainingData[:, 2], trainingLabels)

    trainingData = lda.transform(trainingData)
    testingData = lda.transform(testingData)

    scatter3d(trainingData[:, 0], trainingData[:, 1],trainingData[:, 2], trainingLabels)
    
    return (trainingData, testingData, trainingLabels, testingLabels)

#%%
trainKnn3(*fitLda3(*train_test_split(*unpickleData())))

#%%
fitLda2(*unpickleData())

#%%
def scatter3d(x, y, z, labels):
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, c = labels) 
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.show()

#%%
# todo: wrong!
def fitPca2(data, labels, components=64, size=(32, 32)):
    print(f'data: {data.shape}, labels: {labels.shape}')
    
    data = StandardScaler().fit_transform(data)

    pca = PCA(n_components=components, svd_solver='randomized', whiten=True)
    data = pca.fit_transform(data)

    print(f'total explained variance: {pca.explained_variance_ratio_[:components].sum()}')
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    print(f'pca components: {pca.components_.shape}')

    showImages(pca.components_.reshape(-1, size[0], size[1]), "components", y=0.9)

    return (data, labels)


def fitPca3(trainingData, testingData, trainingLabels, testingLabels, components=64, size=(32, 32)):
    
    print(f'trainingData: {trainingData.shape}, trainingLabels: {trainingLabels.shape}')
    print(f'testingData: {testingData.shape}, testingLabels: {testingLabels.shape}')

    scaler = StandardScaler()
    trainingData = scaler.fit_transform(trainingData)
    testingData = scaler.fit_transform(testingData)

    pca = PCA(n_components=components, svd_solver='randomized', whiten=True)
    pca.fit(trainingData)

    print(f'total explained variance: {pca.explained_variance_ratio_[:components].sum()}')
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    print(f'pca components: {pca.components_.shape}')
    showImages(pca.components_.reshape(-1, size[0], size[1]), "components", y=0.9)

    trainingData = pca.transform(trainingData)
    testingData = pca.transform(testingData)
    
    return (trainingData, testingData, trainingLabels, testingLabels)

fitPca3(*train_test_split(*unpickleData()))

#%%
def trainKnn2(data, labels):
    trainingData, testingData, trainingLabels, testingLabels = train_test_split(data, labels)

    print(f'trainingData: {trainingData.shape}, trainingLabels: {trainingLabels.shape}')
    print(f'testingData: {testingData.shape}, testingLabels: {testingLabels.shape}')

    k = crossValidateKnn(trainingData, trainingLabels)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(trainingData, trainingLabels)
    print(f'training accuracy: {knn.score(trainingData, trainingLabels)}')
    print(f'testing accuracy: {knn.score(testingData, testingLabels)}')

    predictedLabels = knn.predict(testingData)
    print(f'confusion matrix: \n {confusion_matrix(testingLabels, predictedLabels)}')
    print(classification_report(testingLabels, predictedLabels))

def trainKnn3(trainingData, testingData, trainingLabels, testingLabels):

    print(f'trainingData: {trainingData.shape}, trainingLabels: {trainingLabels.shape}')
    print(f'testingData: {testingData.shape}, testingLabels: {testingLabels.shape}')

    k = crossValidateKnn(trainingData, trainingLabels)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(trainingData, trainingLabels)
    print(f'training accuracy: {knn.score(trainingData, trainingLabels)}')
    print(f'testing accuracy: {knn.score(testingData, testingLabels)}')

    predictedLabels = knn.predict(testingData)
    print(f'confusion matrix: \n {confusion_matrix(testingLabels, predictedLabels)}')
    print(classification_report(testingLabels, predictedLabels))

#%%
# trainKnn2(*fitPca2(*unpickleData()))

#%%
trainKnn3(*fitPca3(*train_test_split(*unpickleData())))



#%%
def trainSvm2(data, labels):
    trainingData, testingData, trainingLabels, testingLabels = train_test_split(data, labels)
    print(f'trainingData: {trainingData.shape}, trainingLabels: {trainingLabels.shape}')
    print(f'testingData: {testingData.shape}, testingLabels: {testingLabels.shape}')

    svm = SVM.SVC(gamma='auto')
    svm.fit(trainingData, trainingLabels)

    print(f'training accuracy: {svm.score(trainingData, trainingLabels)}')
    print(f'testing accuracy: {svm.score(testingData, testingLabels)}')

    predictedLabels = svm.predict(testingData)
    print(f'confusion matrix: \n {confusion_matrix(testingLabels, predictedLabels)}')
    print(classification_report(testingLabels, predictedLabels))

def trainSvm3(trainingData, testingData, trainingLabels, testingLabels):
    print(f'trainingData: {trainingData.shape}, trainingLabels: {trainingLabels.shape}')
    print(f'testingData: {testingData.shape}, testingLabels: {testingLabels.shape}')

    svm = SVM.SVC(gamma='auto')
    svm.fit(trainingData, trainingLabels)

    print(f'training accuracy: {svm.score(trainingData, trainingLabels)}')
    print(f'testing accuracy: {svm.score(testingData, testingLabels)}')

    predictedLabels = svm.predict(testingData)
    print(f'confusion matrix: \n {confusion_matrix(testingLabels, predictedLabels)}')
    print(classification_report(testingLabels, predictedLabels))

trainSvm3(*fitPca3(*train_test_split(*unpickleData())))


#%%
def trainMlp3(trainingData, testingData, trainingLabels, testingLabels):
    
    print(f'trainingData: {trainingData.shape}, trainingLabels: {trainingLabels.shape}')
    print(f'testingData: {testingData.shape}, testingLabels: {testingLabels.shape}')

    scaler = StandardScaler()
    trainingData = scaler.fit_transform(trainingData)
    testingData = scaler.fit_transform(testingData)

    mlp = MLPClassifier(random_state=0, verbose=True)
    mlp.fit(trainingData, trainingLabels)

    print(mlp)
    print(f'training accuracy: {mlp.score(trainingData, trainingLabels)}')
    print(f'testing accuracy: {mlp.score(testingData, testingLabels)}')

    predictedLabels = mlp.predict(testingData)
    print(f'confusion matrix: \n {confusion_matrix(testingLabels, predictedLabels)}')
    print(classification_report(testingLabels, predictedLabels))


#%%

trainSvm2(*fitPca2(*unpickleData()))
trainKnn2(*fitPca2(*unpickleData()))
trainKnn2(*fitLda2(*unpickleData()))
trainKnn2(*fitLda2(*fitPca2(*unpickleData())))
trainSvm2(*fitPca2(*unpickleData()))
trainSvm2(*fitLda2(*unpickleData()))
trainSvm2(*fitLda2(*fitPca2(*unpickleData())))

# todo: use pipeline?

#%%
def getData():
    """Return unpicked data seperated into training and testing sets"""
    return train_test_split(*unpickleData(), random_state=0)

#%%
trainKnn3(*getData())

#%%
trainKnn3(*fitLda3(*getData()))


#%%
trainKnn3(*fitPca3(*getData()))

#%%
trainKnn3(*fitLda3(*fitPca3(*getData())))

#%%
trainSvm3(*fitPca3(*getData()))

#%%
trainSvm3(*fitLda3(*fitPca3(*getData())))

#%%
trainMlp3(*getData())

#%%
trainMlp3(*fitLda3(*getData()))


#%%
trainMlp3(*fitPca3(*getData()))

#%%
trainMlp3(*fitLda3(*fitPca3(*getData())))

#%%
directories = [{'path': p, 'count': len(list(p.iterdir())) } for p in path.Path('./cropped').glob('*') if p.is_dir()]
for dir in directories:
    source = dir['path']
    count = dir['count']

    target = source.parent.parent / 'aug' / source.name
    print(source)
    # target.mkdir(parents=True, exist_ok=True)

    p = Augmentor.Pipeline(source_directory=source, output_directory=target.resolve())
    p.skew(probability=0.5)
    p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
    p.rotate_random_90(probability=0.1)
    p.flip_random(probability=0.5)
    p.sample(1)


#%%