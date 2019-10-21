import os
import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from time import time


def load_audio(fn, duration=5, samp_rate=22050):
    '''
	Loads an audio file and splits the audio into segments of length duration 
    with the given sampling rate.
    Returns a 2D NumPy array where each row contains a segment.
	'''
    
    audio, samp_rate = librosa.load(fn, sr=samp_rate)
    seg_len = samp_rate * duration
    num_segs = len(audio) // seg_len
    data = np.zeros((num_segs, seg_len))

    for k in range(num_segs):

        seg = audio[k*seg_len : (k + 1)*seg_len]
        data[k, :] = seg

    return data

def build_data(paths, file_ext='mp3', duration=5, samp_rate=22050):
    '''
    Loads all the files with the file extension in the given list of paths.
    Returns a 2D NumPy array where each row contains a segement.
    '''

    data = np.zeros((0, samp_rate * duration))

    for path in paths:

        print('Processing directory {0}'.format(path))
        fn_list = [fn for fn in os.listdir(path) if fn.endswith('.' + file_ext.lower())]

        for fn in fn_list:

            print('Processing file {0}'.format(fn))
            audio = load_audio(os.path.join(path, fn), duration=duration)
            data = np.vstack((data, audio))

    print('Loaded data with {0} segments and {1:.2f} minutes'.format(
        data.shape[0], data.shape[0]*duration/60))

    return data

def extract_features(signal, n_components=8):
    '''
    Extracts the features for a signal and applies PCA to reduce the number of 
    dimensions by using n_components components of the PCA decomposition.
    '''

    feats = librosa.feature.mfcc(y=signal, n_mfcc=16, n_mels=128)
    feats -= np.mean(feats, axis=0, keepdims=True)
    pca = PCA(n_components=n_components)
    comps, _ = librosa.decompose.decompose(feats, transformer=pca)
    comps = comps.flatten()
    return comps

def load(name):
    '''
    Wrapper around the NumPy load function.
    Returns the contents of a npy file or None if the file doesn't exist or 
    can't be loaded.
    '''

    fn = name + '.npy'
    try:
        rtn = np.load(fn)
        print('Loaded data in {0}'.format(fn))
        return rtn
    except:
        print('Failed to load {0}'.format(fn))
        return None

def check_paths(paths):
    '''
    Checks that the list of paths is valid.
    Returns True if all the paths are valid and False otherwise.
    '''

    for path_group in paths:
        for path in path_group:
            if (not os.path.isdir(path)):
                print('The path {0} is invalid'.format(path))
                return False

    return True

def train_and_test(title, *paths):
    '''
    Preprocesses and extracts features from the given the paths. 
    Trains a linear SVM, RBF SVM, and feedforward network on a training set and 
    validates the models on a testing set.
    Returns a list of the models and a list of the corresponding accuracy of 
    each model in a tuple.
    '''

    if (not check_paths(paths)):
        return None

    num_feats = 128
    feats = load(title + 'features')
    labels = load(title + 'labels')

    if (feats is None or labels is None):

        # Process audio files
        start_time = time()
        data = list()
        for c, path in enumerate(paths):

            c_data = load(title + 'data_c' + str(c))

            if (c_data is None):
                c_data = build_data(path)
                fn = title + 'data_c' + str(c) + '.npy'
                print('Saved data for class {0} to {1}'.format(c, fn))
                np.save(fn, c_data, allow_pickle=False)

            data.append(c_data)
            print('Appended data of length {0}'.format(len(c_data)))

        data_time = (time() - start_time)/60
        print('Processed data in {0:.2f} minutes'.format(data_time))

        # Extract features from audio
        start_time = time()
        feats = np.zeros((0, num_feats))
        labels = np.array(list(), dtype=int)
        for c, dataset in enumerate(data):

            class_feats = np.zeros((0, num_feats))

            for seg in dataset:

                class_feats = np.vstack((class_feats, extract_features(seg)))

            feats = np.vstack((feats, class_feats))
            labels = np.append(labels, c*np.ones(len(class_feats), dtype=int))
            print('Processed features for class {0} of size {1}'.format(c, class_feats.shape))

        print('Saving features of size {0}'.format(feats.shape))
        np.save(title + 'features.npy', feats, allow_pickle=False)
        np.save(title + 'labels.npy', labels, allow_pickle=False)
        feats_time = (time() - start_time)/60
        print('Extracted features of size {0} in {1:.2f} minutes'.format(
            feats.shape, feats_time))

    # Split the features and labels for training and testing sets
    train_feats, test_feats, train_labels, test_labels = train_test_split(
        feats, labels, test_size=0.2, random_state=1, shuffle=True)
    print('Training dataset has {0} features'.format(len(train_labels)))
    print('Testing dataset has {0} features'.format(len(test_feats)))

    # Initialize classifiers
    names = ['Linear SVM', 'RBF SVM', 'ANN']
    classifiers = [
        LinearSVC(verbose=1, multi_class='ovr', tol=1e-6,
            max_iter=int(2e3)),
        SVC(verbose=True, decision_function_shape='ovr', tol=1e-6,
            max_iter=-1),
        MLPClassifier(verbose=False, learning_rate='adaptive', batch_size=32,
            hidden_layer_sizes=num_feats, tol=1e-6, max_iter=int(1e3)),
    ]

    # Train the models and cross validate them
    models = list()
    accs = list()
    for name, clf in zip(names, classifiers):

        print('Training model {0}'.format(name))
        start_time = time()
        model = clf.fit(train_feats, train_labels)
        models.append(model)
        train_time = (start_time - time())/60
        print('Trained model {0} in {1} minutes'.format(name, train_time))

        print('Testing model {0}'.format(name))
        start_time = time()
        acc = clf.score(test_feats, test_labels)
        accs.append(acc)
        test_time = (start_time - time())/60
        print('Tested model {0} in {1} minutes'.format(name, test_time))
        print('Model {0} has accuracy {1}'.format(name, acc))

        fn = title + name + '_model.pkl'
        try:
            with open(fn, 'wb') as file:
                pickle.dump(model, file)
            print('Saved {0} model in {1}'.format(name, fn))
        except:
            print('Failed to save {0} model to {1}'.format(name, fn))

    return models, acc
