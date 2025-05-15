import functools
import numpy as np
from . import config


identity_func = lambda x: x


def apply_boxcox(x, lambda_value):
    if lambda_value == 0:
        return np.log(x)
    else:
        return (x ** lambda_value - 1) / lambda_value


def direction_norm2(vector):
    return (vector + 1) * 0.5 + 0.5


def cesnet_normalize(mode):
    return normalize(mode, "cesnet")


def orange_normalize(mode):
    return normalize(mode, "orange")


def normalize(mode, dataset):
    if dataset == "orange":
        upper_iat = 5
        iat_shift = 0.001
        lambda_value = -2.823583216213916
    elif dataset == "cesnet":
        upper_iat = 5000
        iat_shift = 4
        lambda_value = -0.8377165052139384
        iat_mean = 125.50596531364157
        iat_std_dev = 679.7548075788984
        pktsize_mean = 661.5952298912993
        pktsize_std = 574.5453011502753
    else:
        print("Dataset ", dataset, "is not supported")
        return None        

    if mode == "minmax":
        
        pktsize_minmax = functools.partial(minmax_scale, mini=1, maxi=1460)
        
        iat_minmax = lambda x: minmax_scale(np.clip(x,0,upper_iat), 0, upper_iat)
        
        transforms = [iat_minmax, pktsize_minmax, identity_func]
        
        final_norm = functools.partial(cesnet_threechannel_wrapper, transforms = transforms)
        
        return final_norm
    
    elif mode == "boxcox":
        
        pktsize_minmax = functools.partial(minmax_scale, mini=1, maxi=1460)
        
        reasonable_iat = functools.partial(shift_to_positive_and_bound, shift_value = iat_shift, upper_bound = upper_iat)
        
        apply_boxcox_to_cesnet = functools.partial(apply_boxcox, lambda_value=lambda_value)
        
        transforms = [ lambda x: apply_boxcox_to_cesnet(reasonable_iat(x)), pktsize_minmax, identity_func]
        
        boxcox_normalize = functools.partial(cesnet_threechannel_wrapper, transforms = transforms)
        
        return boxcox_normalize
    
    elif mode == "pktsize-std":
        
        pktsize_std = functools.partial(standardize_vector, mean=pktsize_mean, std_dev=pktsize_std)
        
        transforms = [ identity_func, pktsize_std, identity_func]
        
        final_norm = functools.partial(cesnet_threechannel_wrapper, transforms = transforms)
        
        return final_norm
    
    elif mode == "all-three-n2":
        
        pktsize_minmax = functools.partial(minmax_scale, mini=1, maxi=1460)
        
        reasonable_iat = functools.partial(shift_to_positive_and_bound, shift_value = iat_shift, upper_bound = upper_iat)
        
        apply_boxcox_to_cesnet = functools.partial(apply_boxcox, lambda_value=lambda_value)
        
        transforms = [ lambda x: apply_boxcox_to_cesnet(reasonable_iat(x)), pktsize_minmax, direction_norm2]
        
        final_norm = functools.partial(cesnet_threechannel_wrapper, transforms = transforms)
        
        return final_norm
    
    elif mode == "all-three-box-std-around1":
        
        pktsize_std = functools.partial(standardize_vector, mean=pktsize_mean, std_dev=pktsize_std)
        
        reasonable_iat = functools.partial(shift_to_positive_and_bound, shift_value = iat_shift, upper_bound = upper_iat)
        
        apply_boxcox_to_cesnet = functools.partial(apply_boxcox, lambda_value=lambda_value)
        
        transforms = [ lambda x: apply_boxcox_to_cesnet(reasonable_iat(x)), pktsize_std, direction_norm2]
        
        final_norm = functools.partial(cesnet_threechannel_wrapper, transforms = transforms)
        
        return final_norm
    
    elif mode == "all-three-box-std-origdir":
        
        pktsize_std = functools.partial(standardize_vector, mean=pktsize_mean, std_dev=pktsize_std)
        
        reasonable_iat = functools.partial(shift_to_positive_and_bound, shift_value = iat_shift, upper_bound = upper_iat)
        
        apply_boxcox_to_cesnet = functools.partial(apply_boxcox, lambda_value=lambda_value)
        
        transforms = [ lambda x: apply_boxcox_to_cesnet(reasonable_iat(x)), pktsize_std, identity_func]
        
        final_norm = functools.partial(cesnet_threechannel_wrapper, transforms = transforms)
        
        return final_norm
        
    else:
        print("Normalization mode ", mode, "is not supported")
        return None
    

def standardize_vector(vec, mean=0, std_dev=1):
    return (vec-mean)/std_dev


def cesnet_transform_to_three_channels_padded( ppi ):
    if len( ppi ) >= config.CUTOFF_POINT:
        ppi = ppi[ :config.CUTOFF_POINT, : ]
    
    new_data = np.zeros( ( config.CUTOFF_POINT, 3 ) )
    new_data[ :len( ppi ), 0 ] = ppi[:,0]
    new_data[ :len( ppi ), 1 ] = ppi[:,1]
    new_data[ :len( ppi ), 2 ] = ppi[:,2]
    return new_data


def minmax_scale(s, mini, maxi):
    up =  s - mini
    down = maxi - mini
    return up/down


def shift_to_positive_and_bound(vect, shift_value, upper_bound):
    vect = vect + shift_value
    new_ub = upper_bound + shift_value
    vect = np.where(vect>new_ub, new_ub, vect)
    return vect


def cesnet_switch_and_transform( data, transform = [identity_func, identity_func, identity_func]):
    assert len(transform)==3
    
    data = np.transpose(data)
    new_data = np.zeros( np.shape( data ) )
    
    new_data[ :, 0 ] = transform[0]( data[ :, 0 ] )
    new_data[ :, 1 ] = transform[1]( data[ :, 2 ] )
    new_data[ :, 2 ] = transform[2]( data[ :, 1 ] )

    return new_data


def cesnet_threechannel_wrapper(ppi, transforms):
    return  cesnet_transform_to_three_channels_padded(cesnet_switch_and_transform(ppi, transforms))
