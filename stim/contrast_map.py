from argparse import ArgumentParser
from ConfigParser import SafeConfigParser
import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py

def parse_options(conffile):
    ### parse configuration file
    confparser = SafeConfigParser()
    confparser.read(conffile)
    
    ### parse command line options
    parser = ArgumentParser()
    parser.add_argument("--bg", dest="bgfilename", default=None)
    parser.add_argument("--bgID", dest="bgID",
                         default=confparser.get("Default Options", "bgID"))
    parser.add_argument("--obj", dest="objfilename", default=None)
    parser.add_argument("--objID", dest="objID",
                         default=confparser.get("Default Options", "objID"))
    parser.add_argument("--objSize", dest="objSize", type=int,
                         default=confparser.getint("Default Options", "objSize"))
    parser.add_argument("--objHSize", dest="objHSize", type=int, default=None)
    parser.add_argument("--objVSize", dest="objVSize", type=int, default=None)
    parser.add_argument("--objHPos", dest="objHPos", type=int, default=None)
    parser.add_argument("--objVPos", dest="objVPos", type=int, default=None)
    parser.add_argument("--mapHRes", dest="mapHRes", type=int,
                         default=confparser.getint("Default Options", "mapHRes"))
    parser.add_argument("--mapVRes", dest="mapVRes", type=int,
                         default=confparser.getint("Default Options", "mapVRes"))
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)
    parser.add_argument("--mode", dest="mode",
                        default=confparser.get("Default Options", "mode"))
    parser.add_argument("--HSLWeight", nargs='*', type=float,
                        default=(confparser.getfloat("Default Options", "HWeight"),
                                 confparser.getfloat("Default Options", "SWeight"),
                                 confparser.getfloat("Default Options", "LWeight"),
                                 ))
    parser.add_argument("--map", dest="outfilename", default=None)

    arg = parser.parse_args()
    
    return arg

def resize_object(img_obj, size=None, ratio=None):
    """
    Resize object image according to size or ratio (If size and ratio are both
    specified, size overrides.)
    """
    assert(ratio or size)
    if size:
        ratio = np.sqrt(float(size) / np.sum(img_obj[:,:,3] > 0.5))
    PILimg_obj = Image.fromarray((img_obj * 255).astype(np.uint8))
    new_width = int(PILimg_obj.size[0] * ratio)
    new_height = int(PILimg_obj.size[1] * ratio)
    return np.asarray(PILimg_obj.resize((new_width, new_height), Image.ANTIALIAS), dtype=float) / 255

def rgb2hsl(pixels):
    if pixels.shape[1] != 3:
        raise ValueError("Input array must be of (N, 3) shape. (N: number of pixels)")
    
    rgb = np.asarray(pixels)
    max_rgb = rgb.max(axis=1)
    min_rgb = rgb.min(axis=1)
    delta_rgb = max_rgb - min_rgb
    sum_rgb = max_rgb + min_rgb
    
    # compute lightness
    l = sum_rgb / 2
    
    # compute saturation
    s = np.empty_like(l)
    idx_case0 = np.where(delta_rgb == 0)[0]
    idx_case1 = np.where((delta_rgb != 0) & (l < 0.5))[0]
    idx_case2 = np.where((delta_rgb != 0) & (l >= 0.5))[0]
    s[idx_case0] = 0
    s[idx_case1] = delta_rgb[idx_case1] / sum_rgb[idx_case1]
    s[idx_case2] = delta_rgb[idx_case2] / (2.0 - sum_rgb[idx_case2])
    
    # compute hue
    h = np.empty_like(l)
    r, g, b = rgb.swapaxes(0, 1)
    idx_case0 = np.where(delta_rgb == 0)[0]
    idx_case1 = np.where((delta_rgb != 0) & (max_rgb == r))
    idx_case2 = np.where((delta_rgb != 0) & (max_rgb == g))
    idx_case3 = np.where((delta_rgb != 0) & (max_rgb == b))
    h[idx_case0] = 0
    h[idx_case1] = (g[idx_case1] - b[idx_case1]) / delta_rgb[idx_case1]
    h[idx_case2] = (b[idx_case2] - r[idx_case2]) / delta_rgb[idx_case2] + 2
    h[idx_case3] = (r[idx_case3] - g[idx_case3]) / delta_rgb[idx_case3] + 4
    h = h * 60
    h[h < 0] = h[h < 0] + 360

    return np.array(zip(h, s, l))
    
def RGB_contrast(pixels1, pixels2, eps=1./255, modeparam=None):
    rgbval1 = pixels1.flatten()
    rgbval2 = pixels2.flatten()
    coldiff = np.abs(rgbval1 - rgbval2)
    colmean = np.sum((rgbval1, rgbval2), axis=0) + eps  # eps to avoid division by zero
    return np.mean(coldiff / colmean)

def HSL_contrast(pixels1, pixels2, eps=1./255, modeparam=(2, 1, 2)):
    weight = modeparam
    H_cont = hue_contrast(pixels1, pixels2)
    SL1 = pixels1[:, 1:3]
    SL2 = pixels2[:, 1:3]
    SLdiff = np.abs(SL1 - SL2)
    SLsum = (SL1 + SL2) + eps    # eps to avoid division by zero
    S_cont, L_cont = np.mean(SLdiff / SLsum, axis=0)
    return np.dot(weight, (H_cont, S_cont, L_cont)) / np.sum(weight)

def hue_mismatch(pixels1, pixels2, modeparam=0.5):
    weight = modeparam
    huediff = (pixels1[:, 0] - pixels2[:, 0]) * 2 * np.pi / 360
    mean_mismatch = (1.0 - np.cos(np.angle(np.mean(np.exp(1.0j * huediff))))) / 2
    variance = 1.0 - np.abs(np.mean(np.exp(1.0j * huediff)))
    return weight * mean_mismatch + (1.0 - weight) * variance

def hue_contrast(pixels1, pixels2, modeparam=None):
    huediff = (pixels1[:, 0] - pixels2[:, 0]) * 2 * np.pi / 360
    return np.mean(np.abs(np.angle(np.exp(1.0j * huediff))) / np.pi)

def saturation_contrast(pixels1, pixels2, eps=1./255, modeparam=None):
    sat1 = pixels1[:, 1]
    sat2 = pixels2[:, 1]
    satdiff = np.abs(sat1 - sat2)
    satmean = np.sum((sat1, sat2), axis=0) + eps    # eps to avoid division by zero
    return np.mean(satdiff / satmean)

def lightness_contrast(pixels1, pixels2, eps=1./255, modeparam=None):
    light1 = pixels1[:, 2]
    light2 = pixels2[:, 2]
    lightdiff = np.abs(light1 - light2)
    lightmean = np.sum((light1, light2), axis=0) + eps  # eps to avoid division by zero
    return np.mean(lightdiff / lightmean)

def compare_pixels(pixels1, pixels2, mask=None, mode='RGB', modeparam=None):
    # check the consistency of the arguments
    if pixels1.shape[1] != 3 or pixels2.shape[1] != 3:
        raise ValueError("Input arrays must be of (N, 3) shape. (N: number of pixels)")
    
    if mask == None:
        mask = np.ones(pixels1.shape[0]).astype(bool)
    elif mask.shape[0] != pixels1.shape[0] or mask.shape[0] != pixels2.shape[0]:
        raise ValueError("Size of mask must be identical to that of the 1st dimension of input arrays")
    
    if mode not in ('RGB', 'HSL', 'HLS', 'hue', 'saturation', 'lightness'):
        raise ValueError("mode should be either 'RGB', 'HSL', 'hue', 'saturation', or 'lightness'")

    # set function for pixel comparison
    if mode == 'RGB':
        compfunc = RGB_contrast
    elif mode in ('HSL', 'HLS'):
        compfunc = HSL_contrast
    elif mode == "hue":
#         compfunc = hue_mismatch
        compfunc = hue_contrast
    elif mode == "saturation":
        compfunc = saturation_contrast
    elif mode == "lightness":
        compfunc = lightness_contrast
        
    return compfunc(pixels1[mask], pixels2[mask], modeparam=modeparam)

def compute_contrast_map(img_bg, img_obj, res_map, mode='RGB', modeparam=None, verbose=False):
    obj_shape = img_obj.shape
    bg_shape = img_bg.shape
    ys = np.linspace(0, bg_shape[0] - obj_shape[0] - 1, res_map[0]).astype(int)
    xs = np.linspace(0, bg_shape[1] - obj_shape[1] - 1, res_map[1]).astype(int)
    mask = (img_obj[:, :, 3] >= 0.5).flatten()
    
    pixels_obj = img_obj[:, :, 0:3].reshape(obj_shape[0]*obj_shape[1], 3)
    pixels_bg = img_bg[:, :, 0:3].reshape(bg_shape[0]*bg_shape[1], 3)
    if mode in ("HSL", "HLS", "hue", "saturation", "lightness"):
        pixels_obj = rgb2hsl(pixels_obj)
        pixels_bg = rgb2hsl(pixels_bg)
    
    print "Scanning %d x %d positions on the background..." % (res_map[1], res_map[0])
    
    contmap = np.zeros(res_map)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            pixels_bgpatch = pixels_bg.reshape(bg_shape[0], bg_shape[1], -1)[y:y+obj_shape[0], x:x+obj_shape[1], :].reshape(obj_shape[0]*obj_shape[1], -1)
            contmap[j, i] = compare_pixels(pixels_obj, pixels_bgpatch, mask, mode=mode, modeparam=modeparam)
        if verbose: print "\t%d of %d points done." % ((j+1)*res_map[1], res_map[0]*res_map[1])
        
    print "All done."
    return contmap, xs, ys

def save_map(filename, mapdata, xs, ys, bgfilename, bgsize, objfilename, objsize, mode, modeparam):
    f = h5py.File(filename, 'w')
    f.create_dataset("Map", data=mapdata)
    f.create_dataset("Xs", data=xs)
    f.create_dataset("Ys", data=ys)
    f.create_dataset("BackgroundFile", data=bgfilename)
    f.create_dataset("BackgroundSize", data=bgsize)
    f.create_dataset("ObjectFile", data=objfilename)
    f.create_dataset("ObjectSize", data=objsize)
    f.create_dataset("MapMode", data=mode)
    if mode in ("HSL", "HLS"):
        f.create_dataset("HSLWeight", data=modeparam)
    f.close()

def plot_map(contmap, objID, img_obj, objsize, bgID, img_bg, bgsize, mode, fn_img=None):
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0.07, right=0.96, bottom=0.05, top=0.9)
    plt.subplot(221)
    plt.imshow(img_obj, extent=[bgsize[1]/2-objsize[1]/2, bgsize[1]/2+objsize[1]/2, bgsize[0]/2+objsize[0]/2, bgsize[0]/2-objsize[0]/2])
    plt.xlim(0, bgsize[1])
    plt.ylim(bgsize[0], 0)
    plt.title("Object")

    plt.subplot(222)
    plt.imshow(img_bg)
    plt.title("Background")

    plt.subplot(223)
    plt.imshow(contmap, cmap='jet',
                      extent=[objsize[1]/2, bgsize[1]-objsize[1]/2, bgsize[0]-objsize[0]/2, objsize[0]/2])
    plt.xlim(0, bgsize[1])
    plt.ylim(bgsize[0], 0)
    plt.colorbar().set_label("{0} contrast index".format(mode))
    plt.title("Contrast map")

    plt.subplot(224)
    plt.imshow(img_bg, alpha=0.99)
    plt.imshow(contmap, cmap='jet',
                      extent=[objsize[1]/2, bgsize[1]-objsize[1]/2, bgsize[0]-objsize[0]/2, objsize[0]/2], alpha=0.6)
    plt.xlim(0, bgsize[1])
    plt.ylim(bgsize[0], 0)
    plt.title("Overlay")

    title = "{mode} contrast map: obj{objID} x bg{bgID}".format(mode=mode, bgID=bgID, objID=objID)
    plt.suptitle(title)

    if fn_img:
        plt.savefig(fn_img)
                
    plt.show()

def plot_map_old(contmap, img_obj, img_bg, mode):
    plt.subplot(221)
    plt.imshow(img_obj, extent=[img_bg.shape[1]/2-img_obj.shape[1]/2, img_bg.shape[1]/2+img_obj.shape[1]/2, img_bg.shape[0]/2+img_obj.shape[0]/2, img_bg.shape[0]/2-img_obj.shape[0]/2])
    plt.xlim(0, img_bg.shape[1])
    plt.ylim(img_bg.shape[0], 0)
    plt.title("Object")

    plt.subplot(222)
    plt.imshow(img_bg)
    plt.title("Background")

    plt.subplot(223)
    plt.imshow(contmap, cmap='jet',
                      extent=[img_obj.shape[1]/2, img_bg.shape[1]-img_obj.shape[1]/2, img_bg.shape[0]-img_obj.shape[0]/2, img_obj.shape[0]/2])
    plt.xlim(0, img_bg.shape[1])
    plt.ylim(img_bg.shape[0], 0)
    plt.colorbar().set_label("{0} contrast index".format(mode))
    plt.title("Contrast map")

    plt.subplot(224)
    plt.imshow(img_bg, alpha=0.99)
    plt.imshow(contmap, cmap='jet',
                      extent=[img_obj.shape[1]/2, img_bg.shape[1]-img_obj.shape[1]/2, img_bg.shape[0]-img_obj.shape[0]/2, img_obj.shape[0]/2], alpha=0.6)
    plt.xlim(0, img_bg.shape[1])
    plt.ylim(img_bg.shape[0], 0)
    plt.title("Overlay")

    plt.show()


def main():
    ### parse configuration file
    confparser = SafeConfigParser()
    confparser.read("contrast_map.ini")
    bgfiledir = confparser.get("Paths", "bgfiledir")
    objfiledir = confparser.get("Paths", "objfiledir")

    ### parse command line options
    opt = parse_options("contrast_map.ini")
    
    ### load image files (note: Python accepts only PNG files)
    if opt.bgfilename:
        bgfilename = opt.bgfilename
    else:
        bgfilename = bgfiledir + confparser.get("Filenames", "bgfilename") % opt.bgID
    img_bg = plt.imread(bgfilename)
    print "Background image:\n\t" + bgfilename
    
    if opt.objfilename:
        objfilename = opt.objfilename
    else:
        objfilename = objfiledir + confparser.get("Filenames", "objfilename") % opt.objID
    img_obj = plt.imread(objfilename)
    print "Object image:\n\t" + objfilename
    
    ### if object image does not contain alpha channel, add it
    if img_obj.shape[2] == 3:
        img_obj = np.append(img_obj, np.ones((img_obj.shape[0], img_obj.shape[1], 1.)), axis=2)
    
    ### resize object image
    ratio = []
    if opt.objHSize:
        ratio.append(float(opt.objHSize) / img_obj.shape[1])
    if opt.objVSize:   
        ratio.append(float(opt.objVSize) / img_obj.shape[0])
    if len(ratio) > 0:
        img_obj = resize_object(img_obj, ratio=min(ratio))
    else:
        img_obj = resize_object(img_obj, size=opt.objSize)
    print "Object size:\n\t%d x %d" % (img_obj.shape[1], img_obj.shape[0])
        
    ### set the resolution of the contrast map
    res_map = [opt.mapVRes, opt.mapHRes]
    for i in (0, 1):
        if res_map[i]:
            assert(0 < res_map[i] <= img_bg.shape[i] - img_obj.shape[i])
        else:
            res_map[i] = img_bg.shape[i] - img_obj.shape[i]
    
    ### set mode-specific parameter
    if opt.mode in ("HSL", "HLS"):
        modeparam = opt.HSLWeight
    else:
        modeparam = None

    ### Compute contrast
    if opt.objHPos and opt.objVPos:
        # if object position is given, compute the correlation only at the given
        # position and abort
        x, y = (opt.objHPos, opt.objVPos)
        obj_shape = img_obj.shape
        bg_shape = img_bg.shape
        pixels_obj = img_obj[:, :, 0:3].reshape(obj_shape[0]*obj_shape[1], 3)
        pixels_bg = img_bg[:, :, 0:3].reshape(bg_shape[0]*bg_shape[1], 3)
        pixels_bgpatch = pixels_bg.reshape(bg_shape[0], bg_shape[1], -1)[y:y+obj_shape[0], x:x+obj_shape[1], :].reshape(obj_shape[0]*obj_shape[1], -1)
        if opt.mode in ("HSL", "HLS", "hue", "saturation", "lightness"):
            pixels_obj = rgb2hsl(pixels_obj)
            pixels_bgpatch = rgb2hsl(pixels_bgpatch)
        
        mask = (img_obj[:, :, 3] >= 0.5).flatten()

        print "Object position:\n\t%d , %d" % (x, y)
        print "Contrast index ({0}):\n\t%{1}".format(mode, compare_pixels(pixels_obj, pixels_bgpatch, mask, mode=mode, modeparam=modeparam))
        print
        quit()
    else:
        # otherwise, compute correlation map
        contmap, xs_contmap, ys_contmap = compute_contrast_map(img_bg, img_obj, res_map, mode=opt.mode, modeparam=modeparam, verbose=opt.verbose)
    
    
    ### output result
    if opt.outfilename:
        # if output filename is given, save the correlation map in HDF5 format
        # (together with the coordinates of the sample points and other
        # miscellaneous information)
        save_map(opt.outfilename, contmap, xs_contmap, ys_contmap, bgfilename, img_bg.shape, objfilename, img_obj.shape, opt.mode, modeparam)
    else:
        # otherwise, plot the map in a window
        plot_map(contmap, opt.objID, img_obj, img_obj.shape, opt.bgID, img_bg, img_bg.shape, opt.mode)

    
if __name__ == "__main__":
    main()