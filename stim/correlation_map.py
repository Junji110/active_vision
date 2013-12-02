"""
Compute correlation map:
map of pixel-by-pixel correlation between object image and patches of background image

Created on Sep 27, 2012

Author: Junji Ito (j.ito@fz-juelich.de)
"""
from optparse import OptionParser
from ConfigParser import SafeConfigParser

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from PIL import Image
import h5py

def parse_options(conffile):
    ### parse configuration file
    confparser = SafeConfigParser()
    confparser.read(conffile)
    
    ### parse command line options
    optparser = OptionParser()
    optparser.add_option("--bg", dest="bgfilename", default=None)
    optparser.add_option("--bgID", dest="bgID",
                         default=confparser.get("Default Options", "bgID"))
    optparser.add_option("--obj", dest="objfilename", default=None)
    optparser.add_option("--objID", dest="objID",
                         default=confparser.get("Default Options", "objID"))
    optparser.add_option("--objSize", dest="objSize", type="int",
                         default=confparser.getint("Default Options", "objSize"))
    optparser.add_option("--objHSize", dest="objHSize", type="int", default=None)
    optparser.add_option("--objVSize", dest="objVSize", type="int", default=None)
    optparser.add_option("--objHPos", dest="objHPos", type="int", default=None)
    optparser.add_option("--objVPos", dest="objVPos", type="int", default=None)
    optparser.add_option("--mapHRes", dest="corrMapHRes", type="int",
                         default=confparser.getint("Default Options", "corrMapHRes"))
    optparser.add_option("--mapVRes", dest="corrMapVRes", type="int",
                         default=confparser.getint("Default Options", "corrMapVRes"))
    optparser.add_option("--verbose", dest="verbose", action="store_true", default=False)
    optparser.add_option("--mode", dest="mode",
                         default=confparser.get("Default Options", "corrMapMode"))
    optparser.add_option("--map", dest="outfilename", default=None)
    (opt, _dummy) = optparser.parse_args()
    
    return opt

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


def color_difference(colval1, colval2):
    coldiff = np.abs(colval1 - colval2)
    colmin = np.min((colval1, colval2), axis=0) + 1./255
    return np.mean(coldiff / colmin)

def color_contrast(colval1, colval2):
    coldiff = np.abs(colval1 - colval2)
    colmean = np.sum((colval1, colval2), axis=0) + 1./255
    return np.mean(coldiff / colmean)


def correlate_pixels(patch1, patch2, mask=None, mode='spearman'):
    """
    Compute spearman (or pearson) correlation between RGB pixel values of two
    image patches of identical dimensions
    """
    assert(mode in ('spearman', 'pearson', 'diff', 'cont'))
    
    patch_flat = [x.reshape(x.shape[0]*x.shape[1], x.shape[2]) for x in (patch1, patch2)] 
    
    if mask == None:
        masks = (lambda x: np.ones(x.shape[0]).astype(bool), lambda x: x[:,3] >= 0.5)
        mask = np.all([masks[x.shape[1] == 4](x) for x in patch_flat], axis=0)
    else:
        mask = mask.flatten()
        
    patch_flat = [x[mask][:,0:3].flatten() for x in patch_flat]
        
    if mode == 'spearman':
        corrfunc = lambda x,y: stats.spearmanr(x, y)[0]
    elif mode == 'pearson':
        corrfunc = lambda x,y: stats.pearsonr(x, y)[0]
    elif mode == "diff":
        corrfunc = color_difference
    elif mode == "cont":
        corrfunc = color_contrast
        
    return corrfunc(patch_flat[0], patch_flat[1])


def get_correlation_map(img_bg, img_obj, res_corrmap, mode='spearman', verbose=False):
    """
    Compute correlation map
    """
    obj_shape = img_obj.shape
    bg_shape = img_bg.shape
    ys = np.linspace(0, bg_shape[0] - obj_shape[0] - 1, res_corrmap[0]).astype(int)
    xs = np.linspace(0, bg_shape[1] - obj_shape[1] - 1, res_corrmap[1]).astype(int)
    mask = img_obj[:,:,3] >= 0.5
    
    print "Scanning %d x %d positions on the background..." % (res_corrmap[1], res_corrmap[0])
    
    corrmap = np.zeros(res_corrmap)
    for j,y in enumerate(ys):
        y_fin = y + obj_shape[0]
        for i,x in enumerate(xs):
            x_fin = x + obj_shape[1]
            patch_bg = img_bg[y:y_fin, x:x_fin]
            corrmap[j,i] = correlate_pixels(img_obj, patch_bg, mask, mode=mode)
        if verbose: print "\t%d of %d points done." % ((j+1)*res_corrmap[1], res_corrmap[0]*res_corrmap[1])
        
#    #  another implementation using generator
#    corrmap_gen = (correlate_pixels(img_obj, img_bg[y:y+obj_shape[0], x:x+obj_shape[1]], mask) for y in ys for x in xs)
#    corrmap = np.fromiter(corrmap_gen, np.float32, len(ys)*len(xs))
#    corrmap = corrmap.reshape(res_corrmap)
    
    print "All done."
    return corrmap, xs, ys


def save_correlation_map(filename, corrmap, xs, ys, bgfilename, bgsize, objfilename, objsize):
    f = h5py.File(filename, 'w')
    f.create_dataset("CorrelationMap", data=corrmap)
    f.create_dataset("Xs", data=xs)
    f.create_dataset("Ys", data=ys)
    f.create_dataset("BackgroundFile", data=bgfilename)
    f.create_dataset("BackgroundSize", data=bgsize)
    f.create_dataset("ObjectFile", data=objfilename)
    f.create_dataset("ObjectSize", data=objsize)
    f.close()


def main():
    ### parse configuration file
    confparser = SafeConfigParser()
    confparser.read("correlation_map.ini")
    bgfiledir = confparser.get("Paths", "bgfiledir")
    objfiledir = confparser.get("Paths", "objfiledir")


    ### parse command line options
    opt = parse_options("correlation_map.ini")
    
    
    ### load image files (caution: python only accepts PNG files)
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
        
    
    ### define the resolution of the correlation map
    res_corrmap = [opt.corrMapVRes, opt.corrMapHRes]
    for i in (0,1):
        if res_corrmap[i]:
            assert(0 < res_corrmap[i] <= img_bg.shape[i] - img_obj.shape[i])
        else:
            res_corrmap[i] = img_bg.shape[i] - img_obj.shape[i]
    
    ### Compute correlation
    if opt.objHPos and opt.objVPos:
        # if object position is given, compute the correlation only at the given
        # position and abort
        x, y = (opt.objHPos, opt.objVPos)
        patch_bg = img_bg[y:y+img_obj.shape[0], x:x+img_obj.shape[1]]
        mask = img_obj[:,:,3] >= 0.5
        print "Object position:\n\t%d , %d" % (x, y)
        print "Correlation:\n\t%f" % correlate_pixels(img_obj, patch_bg, mask, opt.mode)
        print
        quit()
    else:
        # otherwise, compute correlation map
        corrmap, xs_corrmap, ys_corrmap = get_correlation_map(img_bg, img_obj, res_corrmap, mode=opt.mode, verbose=opt.verbose)
    
    
    ### output result
    if opt.outfilename:
        # if output filename is given, save the correlation map in HDF5 format
        # (together with the coordinates of the sample points and other
        # miscellaneous information)
        save_correlation_map(opt.outfilename, corrmap, xs_corrmap, ys_corrmap, bgfilename, img_bg.shape, objfilename, img_obj.shape)
    else:
        # otherwise, plot the map in a window
        maptitle = "Correlation Map"
        cblabel = "Correlation"
        if opt.mode == 'diff':
            corrmap = 2 * (1 - (corrmap - corrmap.min()) / (corrmap.max() - corrmap.min())) - 1
            maptitle = "Difference Map"
            cblabel = "Normalized Difference"
        elif opt.mode == 'cont':
            corrmap = 2 * (1 - (corrmap - corrmap.min()) / (corrmap.max() - corrmap.min())) - 1
            maptitle = "Contrast Map"
            cblabel = "Normalized Contrast"
        plt.imshow(corrmap, vmin=-1, vmax=1, cmap='jet',
                          extent=[img_obj.shape[1]/2, img_bg.shape[1]-img_obj.shape[1]/2, img_bg.shape[0]-img_obj.shape[0]/2, img_obj.shape[0]/2])
        plt.xlim(0, img_bg.shape[1])
        plt.ylim(img_bg.shape[0], 0)
        plt.colorbar().set_label(cblabel)
        plt.title(maptitle)
        plt.show()
    
    
if __name__ == "__main__":
    main()