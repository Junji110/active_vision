import os
import re

import scipy.io as spio

def find_filenames(datadir, subject, session, rec, filetype):
    if filetype not in ['imginfo', 'stimtiming', 'param', 'parameter', 'task', 'daq', 'lvd', 'odml', 'hdf5', 'RF']:
        raise ValueError("Filetype {0} is not supported.".format(filetype))

    if filetype in ['daq', 'lvd', 'hdf5', 'odml']:
        searchdir = "{dir}/{sbj}/{sess}".format(dir=datadir, sbj=subject, sess=session)
        re_filename = re.compile('{sess}.*_rec{rec}.*\.{filetype}$'.format(sess=session, rec=rec, filetype=filetype))
    elif filetype in ['RF',]:
        searchdir = "{dir}/{sbj}/{sess}".format(dir=datadir, sbj=subject, sess=session)
        re_filename = re.compile("{0}{1}.*".format(filetype, session))
    else:
        searchdir = "{dir}/{sbj}/{sess}/{sess}_rec{rec}".format(dir=datadir, sbj=subject, sess=session, rec=rec)
        re_filename = re.compile(".*{0}.*".format(filetype))

    filenames = os.listdir(searchdir)
    fn_found = []
    for fn in filenames:
        match = re_filename.match(fn)
        if match:
            fn_found.append("{0}/{1}".format(searchdir, fn))

    if len(fn_found) == 0:
        raise IOError("Files of type '{0}' not found.".format(filetype))
    else:
        return fn_found

def get_imgID(stimdir, stimsetname):
    imgIDs = []
    # for i in range(1, 61):
    i_img = 1
    while True:
        fn_img = "{0}/{1}/{2}.mat".format(stimdir, stimsetname, i_img)
        if os.path.exists(fn_img):
            img_mat = spio.loadmat(fn_img, struct_as_record=False, squeeze_me=True)
            imgIDs.append(img_mat['information'].backgroundid)
            i_img += 1
        else:
            break
    return imgIDs

def get_objID(stimdir, stimsetname):
    objIDs = []
    # for i in range(1, 61):
    i_img = 1
    while True:
        fn_img = "{0}/{1}/{2}.mat".format(stimdir, stimsetname, i_img)
        if os.path.exists(fn_img):
            img_mat = spio.loadmat(fn_img, struct_as_record=False, squeeze_me=True)
            objIDs.extend(img_mat['information'].objectid)
            i_img += 1
        else:
            break
    return sorted(list(set(objIDs)))

if __name__ == "__main__":
    stimdir = "C:/Users/ito/datasets/osaka/stimuli"
    stimsetnames = (
        "fv_psycho2_large",
    )
    for stimsetname in stimsetnames:
        print stimsetname
        print get_imgID(stimdir, stimsetname)
        print get_objID(stimdir, stimsetname)
        print