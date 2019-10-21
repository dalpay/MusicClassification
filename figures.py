import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def display_loss(loss, title):
    '''
    Displays the loss curve.
    '''

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(loss) + 1), loss)
    ax.set(xlabel='Epoch', ylabel='Loss', title=title + ' Loss Evolution')
    fig.tight_layout()

def display_feature_stats(features, labels, classes='', title=''):
    '''
    Displays the mean and standard deviation for the features.
    '''

    unq_labs = np.unique(labels)

    fig, ax = plt.subplots(1, len(unq_labs), sharex=True, sharey=True,
        figsize=(12, 4))

    for k, c in enumerate(unq_labs):

        inds = np.where(labels == c)
        feats = features[inds[0]]
        mean = np.mean(feats, axis=0)
        var = np.var(feats, axis=0)

        ax[k].errorbar(np.arange(1, len(mean) + 1), mean, yerr=var,
            linestyle = None, fmt='.', ecolor='c')
        ax[k].set(xlabel='Feature', ylabel='Feature value',
            title= classes[c] + ' Feature Statistics',  ylim=(-0.1, 0.1))

def display_confusion_matrix(y_true, y_pred, classes, normalize=False,
    title='', cmap=plt.cm.Blues):
    '''
    Based on
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    '''

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label',
           title=title + ' Confusion Matrix')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

def load_models(fn):
    '''
    Wrapper around the Pickle load function.
    '''

    models = None

    try:
        with open(fn, 'rb') as file:
            models = pickle.load(file)
        print('Loaded {0}'.format(fn))
    except:
        print('Failed to load {0}'.format(fn))

    return models


def display_figures(path, title, classes):
    '''
    Displays the feature statistics, loss curve, and confusion matrix based on 
    the features and models in the path.
    '''
    
    features = np.load(path + 'features.npy')
    labels = np.load(path + 'labels.npy')

    display_feature_stats(features, labels, classes=classes, title=title)

    train_feats, test_feats, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=1, shuffle=True)

    names = ['Linear SVM', 'RBF SVM', 'ANN']
    for name in names:

        model = load_models(path + name + '_model.pkl')
        if (model is not None):

            print('Model {0}: '.format(name))
            print('Training loss: {0}'.format(model.score(train_feats, train_labels)))
            print('Testing loss: {0}'.format(model.score(test_feats, test_labels)))

            if (name == 'ANN'):
                display_loss(model.loss_curve_, title=title)

            pred_labels = model.predict(test_feats)
            display_confusion_matrix(test_labels, pred_labels, classes=classes,
                title=title + ' ' + name)

    plt.show()

