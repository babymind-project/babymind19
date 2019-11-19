import cv2
import IPython
import numpy as np
import scipy

DEPTH_MIN = 50
DEPTH_MAX = 20000

def preprocess_img(img, scale = 0.5):
    img = cv2.resize(img,None, fx=scale, fy = scale)
    copyed = np.copy(img)
    copyed = copyed/255.
    return copyed

def preprocess_depth(depth, scale = 0.5):
    depth = cv2.resize(depth, None, fx=scale, fy=scale)
    
    copyed = np.copy(depth)

    nan_mask = np.zeros_like(copyed)
    nan_mask[np.where(np.isnan(copyed))] = 1
    
    copyed[np.where(np.isnan(copyed))] = DEPTH_MIN
    copyed[np.where(np.isinf(copyed))] = DEPTH_MAX
    copyed[np.where(copyed<DEPTH_MIN)] = DEPTH_MIN  
    copyed[np.where(copyed>DEPTH_MAX)] = DEPTH_MAX

    copyed = copyed/1000
    return copyed, nan_mask

def preprocess_mask(mask, scale = 0.5):
    mask_ch = mask.shape[2]
    
    mask = cv2.resize(mask, None, fx=scale, fy=scale)
    if mask_ch == 1:
        mask = np.expand_dims(mask, 2)
    mask[np.where(mask>0.5)] = 1
    mask[np.where(mask<=0.5)] = 0

    mask_filled = []
    for ch in range(mask_ch):
        mask_filled_ch = scipy.ndimage.binary_fill_holes(np.asarray(mask[:,:,ch], dtype = np.int32))
        mask_filled.append(np.expand_dims(mask_filled_ch, 2))
    mask_filled = np.concatenate(mask_filled, 2)
    return np.asarray(mask_filled, dtype = np.float32)


def preprocess_bbox(pc, mask, scale=0.5):
    k = 0.05 # pick top k% and bottom k% 
    mask_ch = mask.shape[2]
    #mask.shape [h,w,ch] 
    #pc.shape  [h,w,3]
    
    pc = np.nan_to_num(pc)
    
    bbox = []
    for ch in range(mask_ch):
        pc_masked = pc[np.where(mask[:,:,ch]>0.5)]
        num = len(pc_masked)

        if len(pc_masked) == 0:
            assert False
        else:
            min_pick = int(num*k)-1
            max_pick = num - int(num*k)
        
            x_sort = np.argsort(pc_masked[:,0])
            y_sort = np.argsort(pc_masked[:,1])
            z_sort = np.argsort(pc_masked[:,2]) 

            x_min = pc_masked[x_sort[min_pick],0]/1000 - 5e-2
            x_max = pc_masked[x_sort[max_pick],0]/1000 + 5e-2

            y_min = pc_masked[y_sort[min_pick],1]/1000 - 5e-2
            y_max = pc_masked[y_sort[max_pick],1]/1000 + 5e-2

            z_min = pc_masked[z_sort[min_pick],2]/1000
            z_max = pc_masked[z_sort[max_pick],2]/1000 + 3e-1

            bbox.append(np.reshape([x_max, y_max, z_max, x_min, y_min, z_min], [1,6]))
    bbox = np.concatenate(bbox, 0)
    return bbox
    
        


            

