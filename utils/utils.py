import numpy as np
from .data_generator import *


def scheduler2_factory(epoch_list, lr_list):
    epoch_list = np.cumsum(epoch_list)
    
    def scheduler2(epoch, lr):
        print("epoch and initial lr in scheduler: ", epoch, lr)
        for i, ep in enumerate(epoch_list):
            if epoch <= ep:
                lr = lr_list[i]
                break
        print("learning rate should now be ", lr)
        return lr
    
    return scheduler2


def scheduler3(epoch, lr):
    if epoch != 0 and epoch % 4 == 0:
        new_lr = lr/2
    else:
        new_lr = lr  
    print("learning rate should now be ", new_lr)
    return new_lr


def get_data_generator(train_x, train_y, val_x, val_y, n_classes, norm_func, 
                       feature_select, batch_size, dim, n_channels):
    training_generator = DataGenerator(list_IDs = train_x, labels = train_y, n_classes = n_classes, 
                                    norm_func=norm_func, feature_select=feature_select,
                                    batch_size = batch_size, dim = dim, n_channels = n_channels)
    validation_generator = DataGenerator(list_IDs = val_x, labels = val_y, n_classes = n_classes, 
                                    norm_func=norm_func, feature_select=feature_select,
                                    batch_size = batch_size, dim = dim, n_channels = n_channels)
    
    return training_generator, validation_generator
