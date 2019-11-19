import IPython
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon



def apply_mask(img, mask_img, alpha=0.5):
    """Apply the given mask to the image.
    """
    masked_img = (1-alpha)*img + alpha*mask_img
    return masked_img


def multichannel_to_image(mask):
    # visualize multichannel image to 3-channel image using colormap
    # each of tab20,tab20b,tab20c contains 20 colors
    cmap0= plt.get_cmap('tab20')
    cmap1 = plt.get_cmap('tab20b')
    cmap2 = plt.get_cmap('tab20c')
    mask_h, mask_w, mask_ch = mask.shape
    img = np.zeros((mask_h, mask_w, 3))
    for ch in range(mask_ch-1, -1, -1):
        if ch<20:
            color = cmap0(ch)
        elif ch<40:
            color = cmap1(ch-20)
        elif ch<60:
            color = cmap2(ch-40)
        colored_mask = mask_ch*np.array(color[:3]) * np.expand_dims(mask[:,:,ch], 2)
        #replace_idx = np.where( img==0 )
        #img[replace_idx] = colored_mask[replace_idx]
        img += colored_mask
    img = img/mask_ch
    img = np.clip(img, -1, 1)
    
    return img


def sigmoid(x):
    sigm = 1./(1.+np.exp(-x))
    return sigm


def random_colors(N, bright = True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

