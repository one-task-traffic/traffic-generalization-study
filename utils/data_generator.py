from tensorflow.keras.utils import to_categorical
import keras
import numpy as np
from .normalizer import *
from .cesnetpreprocessing import *
from . import config

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, n_classes, norm_func = normalize("boxcox", "cesnet"), 
                 feature_select=[0,1,2], batch_size=32, dim=(config.CUTOFF_POINT), n_channels=3):
        'Initialization'
        super().__init__(use_multiprocessing=True, workers=16)
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm_func = norm_func
        self.feature_select = feature_select

        # sort ID list without messing the label association up
        self.labels = labels
        self.list_IDs = list_IDs
        self.__sort_IDs()
        self.on_epoch_end()
        
    def __sort_IDs(self):
        labels = self.labels
        list_IDs = self.list_IDs
        
        assert(len(labels) == len(list_IDs))
        
        id_label_comb = [[list_IDs[k],labels[k]] for k in range(len(list_IDs))]
        'Sorts list by ID'
        sorted_id_label = sorted(id_label_comb,key=lambda i:i[0])
        self.list_IDs = [b[0] for b in sorted_id_label]
        self.labels = [b[1] for b in sorted_id_label]
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        y = [self.labels[k] for k in indexes]
        
        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X, to_categorical(y, num_classes=self.n_classes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialize the final batch of this iteration

        # list of the data points for IDs in list_IDs_temp
        X_batch = load_data_for_ids(list_IDs_temp)
        
        # Initialize the final batch of this iteration
        X = np.empty((self.batch_size, self.dim, self.n_channels))

        for i, raw_x in enumerate(X_batch):
            # creates a normalized data points with the correct shapes and puts it in place in the batch 
            X[i,] = self.norm_func(raw_x)[:,self.feature_select]
            
        return X
