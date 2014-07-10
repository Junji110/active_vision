import os
import re

import scipy.io as spio

def find_filenames(datadir, subject, session, rec, filetype):
    if filetype not in ['imginfo', 'stimtiming', 'param', 'parameter', 'task', 'daq', 'lvd', 'odml', 'hdf5']:
        raise ValueError("Filetype {0} is not supported.".format(filetype))
    
    if filetype in ['daq', 'lvd', 'hdf5', 'odml']:
        searchdir = "{dir}/{sbj}/{sess}".format(dir=datadir, sbj=subject, sess=session)
        re_filename = re.compile('{sess}.*_rec{rec}.*\.{filetype}$'.format(sess=session, rec=rec, filetype=filetype))
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
    print stimsetname
    for i in range(1, 49):
        fn_img = "{0}/{1}/{2}.mat".format(stimdir, stimsetname, i)
        img_mat = spio.loadmat(fn_img, struct_as_record=False, squeeze_me=True)
        print "{0},".format(img_mat['information'].backgroundid),


if __name__ == "__main__":
    stimdir = "C:/Users/ito/datasets/osaka/stimuli"
#     stimsetname = "fv_random_3"
    stimsetname = "fv_random_5_large_stripes"
    get_imgID(stimdir, stimsetname)