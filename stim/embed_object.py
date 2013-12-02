"""
Embed an object in a background image

Created on Oct 08, 2012

Author: Junji Ito (j.ito@fz-juelich.de), Serge Strokov (s.strokov@fz-juelich.de)
"""
from optparse import OptionParser
from ConfigParser import SafeConfigParser

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def parse_options(conffile):
    ### parse configuration file
    confparser = SafeConfigParser()
    confparser.read("correlation_map.ini")

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
    optparser.add_option("--objHPos", dest="objHPos", type="int",
                         default=confparser.getint("Default Options", "objHPos"))
    optparser.add_option("--objVPos", dest="objVPos", type="int",
                         default=confparser.getint("Default Options", "objVPos"))
    optparser.add_option("--trans", dest="transparency", type="float",
                         default=confparser.getfloat("Default Options", "transparency"))
    optparser.add_option("--iadjust", action="store_true", dest="iadjust",
                         default=confparser.getboolean("Default Options", "iadjust"))
    optparser.add_option("--img", dest="outfilename", default=None)
    (opt, _dummy) = optparser.parse_args()
    
    return opt
    
def resize_object(img_obj, size=None, ratio=None):
    """
    Resize object image according to size or ratio
    (If size and ratio are both specified, size overrides.)
    """
    assert(ratio or size)
    if size:
        ratio = np.sqrt(float(size) / np.sum(img_obj[:,:,3] > 0.5))
    PILimg_obj = Image.fromarray((img_obj * 255).astype(np.uint8))
    new_width = int(PILimg_obj.size[0] * ratio)
    new_height = int(PILimg_obj.size[1] * ratio)
    return np.asarray(PILimg_obj.resize((new_width, new_height), Image.ANTIALIAS), dtype=float) / 255


def adjust_intensity(img, img_ref, alpha_threshold=0.5):
    """
    Adjust intensity distribution of a given image to that of a reference image
    """
    fg = np.array(img)
    bg = np.array(img_ref)
    
    fg_flat = fg.reshape(fg.shape[0] * fg.shape[1], fg.shape[2])
    bg_flat = bg.reshape(bg.shape[0] * bg.shape[1], bg.shape[2])
    
    ### calculate intensities     
    def get_intensity(img):
        if img.shape[1] == 3:
            img_tmp = img
        else:
            img_tmp = img[img[:,3] > alpha_threshold]
        return np.array(map(max, img_tmp[:,0:3]))
    I_fg = get_intensity(fg_flat)
    I_bg = get_intensity(bg_flat)
    
    I_bg.sort()
    argsort_I_fg = I_fg.argsort() # sorting indices
    len_ratio = len(I_bg)/float(len(I_fg)) # calculating length ratio
    
    ### obtaining adjusted intensity values
    I_adjust = np.empty(len(argsort_I_fg))
    for i,idx in enumerate(argsort_I_fg):
        idx_ini = int(len_ratio*i)
        idx_fin = int(len_ratio*(i+1))
        if idx_ini == idx_fin:
            I_adjust[idx] = I_bg[idx_ini]    
        else:
            I_adjust[idx] = np.mean(I_bg[idx_ini:idx_fin])    
    
    #normalize pixel values of the object
    j=0
    for i in range(len(fg_flat)):
        if fg_flat[i][3] > alpha_threshold:
            if I_fg[j] == 0:
                fg_flat[i][0:3] = I_adjust[j]
            else:
                fg_flat[i][0:3] = fg_flat[i][0:3] * (I_adjust[j] / I_fg[j])
            j=j+1
            
#    ### check results    
#    obj_orig=img
#    obj=fg
#    I_obj_orig = get_intensity(obj_orig.reshape(obj_orig.shape[0]*obj_orig.shape[1], 4))
#    I_obj_matched = get_intensity(obj.reshape(obj.shape[0]*obj.shape[1], 4))
#    I_bg = get_intensity(bg.reshape(bg.shape[0]*bg.shape[1], 3))
#    
#    I_obj_orig.sort()
#    I_obj_matched.sort()
#    I_bg.sort()
#    
#    plt.figure(0)
#    plt.subplot(311)
#    plt.plot(I_obj_orig, label='Original')
#    plt.xlim(0, len(I_obj_orig))
#    plt.legend(loc=4)
#    plt.subplot(312)
#    plt.plot(I_obj_matched, label='Normalized')
#    plt.xlim(0, len(I_obj_matched))
#    plt.legend(loc=4)
#    plt.subplot(313)
#    plt.plot(I_bg, label='Background')
#    plt.xlim(0, len(I_bg))
#    plt.legend(loc=4)
#    
#    ### SHOW OBJECTS
#    plt.figure(1)
#    plt.imshow(obj_orig)
#    plt.title("original object")
#    
#    plt.figure(2)
#    plt.imshow(obj)
#    plt.title("normalized object")
#    
#    plt.show()
    
    return fg


def embed_object(img_bg, img_obj, pos_obj, transparency=1.0, iadjust=False):
    """
    Embed an object image in a background image at a given position
    """
    assert(0. <= transparency <= 1.)
    
    img_obj_local = np.array(img_obj)
    
    ### scale alpha channel values by the given transparency
    img_obj_local[:,:,3] *= transparency
    
    ### embed the object to the background
    img_embed = np.array(img_bg)
    for y in range(img_obj.shape[0]):
        for x in range(img_obj.shape[1]):
            alpha = img_obj_local[y,x,3]
            img_embed[pos_obj[0] + y, pos_obj[1] + x, 0:3] = alpha * img_obj_local[y, x, 0:3] + (1. - alpha) * img_bg[pos_obj[0] + y, pos_obj[1] + x, 0:3]
            
    return img_embed


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
    
    if opt.objfilename:
        objfilename = opt.objfilename
    else:
        objfilename = objfiledir + confparser.get("Filenames", "objfilename") % opt.objID
    img_obj = plt.imread(objfilename)
    
    
    ### if object image does not contain alpha channel, add it
    if img_obj.shape[2] == 3:
        img_obj = np.append(img_obj, np.ones((img_obj.shape[0], img_obj.shape[1], 1)), axis=2)
    
    
    ### resize object image
    ratio = []
    if opt.objHSize:
        ratio.append(float(opt.objHSize) / img_obj.shape[1])
    if opt.objVSize:   
        ratio.append(float(opt.objVSize) / img_obj.shape[0])
    if len(ratio) > 0:
        img_obj_resize = resize_object(img_obj, ratio=min(ratio))
    else:
        img_obj_resize = resize_object(img_obj, size=opt.objSize)
        
    
    ### set the top-left corner of the embedding area
    pos_obj = [opt.objVPos, opt.objHPos]
    assert(0<= pos_obj[0] < img_bg.shape[0]-img_obj_resize.shape[0])
    assert(0<= pos_obj[1] < img_bg.shape[1]-img_obj_resize.shape[1])
    
    
    ### adjust the object intensity to the local background intensity
    if opt.iadjust:
        patch_bg = img_bg[pos_obj[0]:pos_obj[0] + img_obj.shape[0], pos_obj[1]:pos_obj[1] + img_obj.shape[1]]
        img_obj = adjust_intensity(img_obj, patch_bg)
        if len(ratio) > 0:
            img_obj_resize = resize_object(img_obj, ratio=min(ratio))
        else:
            img_obj_resize = resize_object(img_obj, size=opt.objSize)
        
    
    ### embed the object
    img_embed = embed_object(img_bg, img_obj_resize, pos_obj, transparency=opt.transparency, iadjust=opt.iadjust)


    ### print summary    
    print
    print "Background image file:\n\t%s" % bgfilename
    print "Object image file:\n\t%s" % objfilename
    print "Object size:\n\t%d x %d" % (img_obj_resize.shape[1], img_obj_resize.shape[0])
    print "Object position:\n\t%d , %d" % (pos_obj[1], pos_obj[0])
    print "Object transparency:\n\t%f" % opt.transparency
    print "Object brightness adjustment:\n\t%s" % ('True' if opt.iadjust else 'False')
    print
    
    
    ### save the obtained image in a file
    if opt.outfilename:
        img_PIL = Image.fromarray((img_embed * 255).astype(np.uint8))
        img_PIL.save(opt.outfilename, format="png")
    else:
        plt.imshow(img_embed)
        plt.show()
    
    
    
    
if __name__ == "__main__":
    main()