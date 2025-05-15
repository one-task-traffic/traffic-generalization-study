import pickle as pkl

file_prefix = "./"

home_path =  "./"
idlist_path = "./"
startid_path = "./"

file_step_size = 16*1024
picked_fine_classes_25 = [4, 7, 16, 26, 42, 47, 53, 56, 62, 70, 78, 82, 84, 96, 106, 110, 114, 120, 122, 126, 127, 150, 168, 169, 182]
fine2coarse = {183: 6, 13: 14, 17: 8, 6: 18, 81: 15, 150: 9, 70: 15, 116: 9, 56: 10, 68: 15, 110: 9, 100: 8, 176: 14, 97: 20, 167: 20, 162: 14, 0: 2, 123: 17, 76: 2, 26: 9, 2: 6, 179: 2, 40: 14, 181: 0, 88: 0, 57: 14, 30: 4, 49: 6, 192: 15, 168: 7, 89: 10, 189: 14, 93: 6, 132: 0, 197: 5, 1: 20, 193: 14, 39: 9, 149: 4, 86: 15, 108: 15, 119: 15, 10: 6, 109: 20, 103: 6, 126: 2, 143: 15, 44: 14, 133: 19, 151: 2, 190: 4, 137: 18, 14: 6, 12: 14, 171: 5, 34: 13, 71: 9, 120: 14, 11: 14, 146: 11, 154: 11, 94: 20, 177: 7, 5: 10, 101: 20, 15: 14, 174: 14, 65: 6, 102: 14, 161: 6, 61: 20, 77: 14, 125: 0, 63: 14, 187: 17, 166: 14, 47: 14, 42: 17, 58: 4, 73: 6, 37: 4, 141: 2, 185: 14, 96: 12, 20: 10, 128: 13, 38: 20, 64: 17, 182: 15, 98: 8, 79: 17, 53: 17, 130: 12, 152: 0, 184: 2, 19: 13, 85: 17, 27: 9, 175: 11, 160: 0, 158: 17, 74: 14, 91: 17, 43: 12, 170: 14, 90: 2, 117: 15, 155: 3, 18: 15, 140: 11, 122: 1, 127: 5, 92: 6, 144: 6, 62: 3, 41: 13, 75: 2, 134: 14, 8: 14, 78: 17, 188: 3, 136: 0, 172: 20, 67: 14, 104: 12, 115: 19, 105: 2, 22: 15, 4: 16, 113: 11, 148: 8, 9: 14, 59: 9, 99: 14, 191: 4, 194: 0, 28: 6, 82: 14, 157: 14, 139: 12, 80: 11, 159: 2, 50: 8, 163: 14, 24: 12, 29: 2, 107: 0, 16: 16, 153: 2, 135: 0, 35: 14, 169: 16, 147: 2, 196: 14, 3: 20, 83: 7, 111: 3, 138: 15, 195: 6, 173: 14, 112: 1, 66: 0, 69: 11, 72: 7, 31: 14, 48: 20, 114: 16, 33: 3, 54: 7, 121: 0, 60: 14, 55: 10, 87: 14, 118: 17, 32: 15, 106: 13, 156: 4, 165: 20, 129: 10, 124: 3, 131: 2, 142: 15, 23: 5, 36: 16, 95: 7, 164: 15, 178: 20, 52: 0, 180: 17, 186: 15, 45: 20, 84: 1, 51: 5, 145: 14, 25: 6, 46: 15, 7: 16, 21: 9}
CUTOFF_POINT = 30
__all_labelled_IDs = None
__both_label_indices = None
__file_startIds = None


def get_all_labelled_IDs():
    global __both_label_indices
    global __all_labelled_IDs 
    if __all_labelled_IDs is not None:
        return __all_labelled_IDs 
        
    with open(idlist_path, 'rb') as file:
        __all_labelled_IDs = pkl.load(file)

    __both_label_indices = all_labelled_IDs["both_label_indices"]
    __all_labelled_IDs = all_labelled_IDs["all_labeled_ids"]
    
    return __all_labelled_IDs


def get_both_label_indices():
    
    global __both_label_indices
    global __all_labelled_IDs 
    
    if __both_label_indices is not None:
        return __both_label_indices 
        
    with open(idlist_path, 'rb') as file:
        __all_labelled_IDs = pkl.load(file)

    __both_label_indices = __all_labelled_IDs["both_label_indices"]
    __all_labelled_IDs = __all_labelled_IDs["all_labeled_ids"]
    
    return __both_label_indices


def get_file_startIds():
    global __file_startIds
    if __file_startIds is not None:
        return __file_startIds

    with open(startid_path, 'rb') as file:
        file_startIds = pkl.load(file)

    print(len(file_startIds))
    file_startIds = sorted(file_startIds, key=lambda x: x[1])
    file_startIds = [(x,y-1) for x,y in file_startIds]
    __file_startIds = file_startIds
    
    return __file_startIds


def map_label_num_to_soori_label(label_num):
    return picked_fine_classes_25.index(label_num)
