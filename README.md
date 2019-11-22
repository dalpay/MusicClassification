# Music Classification
This repo contains modules for music classification. 
# Musical Instrument Classification
This repo contains modules for music classification tasks. The file 
`audio_classify.py` contains functions to preprocess audio files, extract 
features, and train classifiers. The file `figures.py` contains functions to 
display figures about the features and classifiers. These functions were applied 
to three music classification tasks in the scripts `classify_bands.py`, 
`classify_genres.py`, and `classify_jazz.py`. 

## Approach
The audio files are first downsampled and split into 5 second segments. The 
MFCCs are computed on the segments and PCA is applied to MFCCs to reduce the 
dimensionality of the features. Three classifiers are then trained on the 
features: linear SVMs, RBF SVMs, and feedforward neural networks. The 
classifiers are then compared by displaying their confusion matrix, loss 
evolution, and performance metrics. 

## Dependencies
- NumPy
- Matplotlib
- Librosa
- Scikit-Learn
