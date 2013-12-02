"""
Return a position for object embedding

Created on Oct 09, 2012

Author: Junji Ito (j.ito@fz-juelich.de)
"""
from optparse import OptionParser
from ConfigParser import SafeConfigParser

import numpy as np
import h5py
import matplotlib.pyplot as plt


def parse_options(conffile):
    ### parse configuration file
    confparser = SafeConfigParser()
    confparser.read(conffile)

    ### parse command line options
    optparser = OptionParser()
    optparser.add_option("--map", dest="corrmapfilename", default=None)
    optparser.add_option("--mode", dest="mode",
                         default=confparser.get("Default Options", "embedPosMode"))
    optparser.add_option("--frac", dest="fraction", type="float",
                         default=confparser.getfloat("Default Options", "fraction"))
    optparser.add_option("--lowth", dest="lowth", type="float", default=None)
    optparser.add_option("--upth", dest="upth", type="float", default=None)
    optparser.add_option("--plot", dest="plot", action="store_true", default=False)
    optparser.add_option("--pmap", dest="probmapfilename", default=None)
    (opt, _dummy) = optparser.parse_args()
    
    return opt
    
def normalize_diffmap(x):
    return 2. * (1. - x / 255.) - 1.

def normalize_contmap(x):
    return 2. * (1. - x) - 1.

def get_position(corrmap, xs, ys, mode="rand", frac=0.01, lowth=None, upth=None):
    probmap = None
    if mode in ("max", "dmax", "cmax"):
        idx = np.unravel_index(corrmap.argmax(), corrmap.shape)
        x = xs[idx[1]]
        y = ys[idx[0]]
    elif mode in ("min", "dmin", "cmin"):
        idx = np.unravel_index(corrmap.argmin(), corrmap.shape)
        x = xs[idx[1]]
        y = ys[idx[0]]
    elif mode in ("prob", "dprob", "cprob"):
        corrmap_flat = corrmap.reshape(corrmap.shape[0] * corrmap.shape[1])
        if lowth!=None and upth!=None:
            assert(lowth < upth)
            assert(lowth <= corrmap_flat.max())
            assert(upth >= corrmap_flat.min())
            lower_threshold = lowth
            upper_threshold = upth
        else:
            argsort_corrmap = corrmap_flat.argsort()
            if mode == 'prob':
                lower_threshold = corrmap_flat[argsort_corrmap[int((1. - frac)*len(corrmap_flat))]]
                upper_threshold = corrmap_flat.max()
            else:
                lower_threshold = corrmap_flat.min()
                upper_threshold = corrmap_flat[argsort_corrmap[int(frac*len(corrmap_flat))]]
        if mode == 'dprob':
            corrmap_flat = normalize_diffmap(corrmap_flat)
            lower_threshold, upper_threshold = map(normalize_diffmap, (upper_threshold, lower_threshold))
        elif mode == 'cprob':
            corrmap_flat = normalize_contmap(corrmap_flat)
            lower_threshold, upper_threshold = map(normalize_contmap, (upper_threshold, lower_threshold))
        probmap_flat = np.array(map(lambda x: x - lower_threshold if lower_threshold<=x<=upper_threshold else 0, corrmap_flat))
        probmap_flat /= sum(probmap_flat)
        probmap = probmap_flat.reshape(corrmap.shape) 
        
        randnum = np.random.rand(1)
        cumprob = 0.
        for i in range(len(probmap_flat)):
            cumprob += probmap_flat[i]
            if cumprob > randnum:
                idx = np.unravel_index(i, corrmap.shape)
                x = xs[idx[1]]
                y = ys[idx[0]]
                break
    else:
        x = xs[np.random.randint(len(xs))]
        y = ys[np.random.randint(len(ys))]
    
    return x, y, probmap

def plot_position(x, y, f, probmap=None, mode='rand'):
    fig = plt.figure(1)
    ax_bg = fig.add_subplot(221)
    ax_obj = fig.add_subplot(222)
    ax_map = fig.add_subplot(223, sharex=ax_bg, sharey=ax_bg)
    ax_pos = fig.add_subplot(224, sharex=ax_bg, sharey=ax_bg)
    
    corrmap = f['CorrelationMap'].value
    maptitle = "Correlation Map"
    if mode in ('dmin', 'dmax', 'dprob'):
        corrmap = normalize_diffmap(corrmap)
        maptitle = "Difference Map"
    elif mode in ('cmin', 'cmax', 'cprob'):
        corrmap = normalize_contmap(corrmap)
        maptitle = "Contrast Map"
    img_bg = plt.imread(f['BackgroundFile'].value)
    img_obj = plt.imread(f['ObjectFile'].value)
    bgshape = np.array(f['BackgroundSize'].value)
    objshape = np.array(f['ObjectSize'].value)
    
    ax_bg.imshow(img_bg)
    ax_bg.set_title("Background")
    
    ax_obj.imshow(img_obj, extent=[-objshape[1]/2, objshape[1]/2, -objshape[0]/2, objshape[0]/2])
    ax_obj.set_xlim(-bgshape[1]/2, bgshape[1]/2)
    ax_obj.set_ylim(-bgshape[0]/2, bgshape[0]/2)
    ax_obj.set_title("Object")
    
    im = ax_map.imshow(corrmap, vmin=-1, vmax=1, cmap='jet', extent=[objshape[1]/2, f['Xs'][-1]+objshape[1]/2, f['Ys'][-1]+objshape[0]/2, objshape[0]/2])
#    plt.colorbar(im, ax=ax_map)
    ax_map.set_title(maptitle)
    
    if probmap == None:
        map_to_plot = corrmap
        (vmin, vmax, cmap) = (-1, 1, 'jet')
        gridcolor = 'black'
    else:
        map_to_plot = probmap
        (vmin, vmax, cmap) = (0, probmap.max(), 'hot')
        gridcolor = 'white'
    im = ax_pos.imshow(map_to_plot, vmin=vmin, vmax=vmax, cmap=cmap, extent=[objshape[1]/2, f['Xs'][-1]+objshape[1]/2, f['Ys'][-1]+objshape[0]/2, objshape[0]/2])
#    plt.colorbar(im, ax=ax_point)
    ax_pos.plot(x+objshape[1]/2, y+objshape[0]/2, 'x', color='gray', ms=10, mew=2)
    ax_pos.set_title("Embedding Position")
    
    ax_bg.set_xlim(0, bgshape[1])
    ax_bg.set_ylim(bgshape[0], 0)
    ax_bg.set_xticks(range(0, bgshape[1], objshape[1]))
    ax_bg.set_xticklabels('')
    ax_bg.set_yticks(range(0, bgshape[0], objshape[0]))
    ax_bg.set_yticklabels('')
    ax_bg.grid()
    ax_map.grid()
    ax_pos.grid(color=gridcolor)
    
    plt.show()
    

def save_probability_map(filename, probmap, xs, ys, bgfilename, bgsize, objfilename, objsize, frac, lowth, upth):
    f = h5py.File(filename, 'w')
    f.create_dataset("ProbabilityMap", data=probmap)
    f.create_dataset("Xs", data=xs)
    f.create_dataset("Ys", data=ys)
    f.create_dataset("BackgroundFile", data=bgfilename)
    f.create_dataset("BackgroundSize", data=bgsize)
    f.create_dataset("ObjectFile", data=objfilename)
    f.create_dataset("ObjectSize", data=objsize)
    f.create_dataset("Fraction", data=frac)
    f.create_dataset("LowerThreshold", data=lowth)
    f.create_dataset("UpperThreshold", data=upth)
    f.close()


def main():
    ### parse command line options
    opt = parse_options("correlation_map.ini")
    
    if opt.corrmapfilename:
        f = h5py.File(opt.corrmapfilename, 'r')
    else:
        print "Error: specify a HDF5 file containing correlation map data."
        quit()
    
    x, y, probmap = get_position(f['CorrelationMap'].value, f['Xs'].value, f['Ys'].value, mode=opt.mode, frac=opt.fraction, lowth=opt.lowth, upth=opt.upth)
        
    print x, y
    
    if opt.probmapfilename:
        if opt.mode in ('prob', 'dprob', 'cprob'):
            if opt.lowth == None and opt.upth==None:
                lower_threshold = 0.
                upper_threshold = 0.
            else:
                lower_threshold = opt.lowth
                upper_threshold = opt.upth
            save_probability_map(opt.probmapfilename, probmap, f['Xs'].value, f['Ys'].value, f['BackgroundFile'].value, f['BackgroundSize'].value, f['ObjectFile'].value, f['ObjectSize'].value, opt.fraction, lower_threshold, upper_threshold)
        else:
            print "Error: operation mode must be either 'prob' or 'dprob'."
            
    if opt.plot:
        if 'BackgroundFile' not in f.keys():
            print "Error: correlation map datafile does not contain sufficient data for plotting."
        else:
            plot_position(x, y, f, probmap, mode=opt.mode)
        
            

if __name__ == "__main__":
    main()