import os

def find_filenames(datadir, subject, session, rec, filetype):
    if filetype not in ['imginfo', 'stimtiming', 'param', 'parameter', 'task', 'daq', 'lvd', 'odml']:
        raise ValueError("filetype must be either of 'imginfo', 'stimtiming', 'param', 'parameter', 'task', or 'daq'.")
    
    # identify the names of metadata files
    if filetype in ['daq', 'lvd', 'odml']:
        searchdir = "{dir}/{sbj}/{sess}".format(dir=datadir, sbj=subject, sess=session)
        if filetype in ['daq', 'lvd']:
            searchtoken = 'rec{0}_pc'.format(rec)
        else:
            searchtoken = filetype
    else:
        searchdir = "{dir}/{sbj}/{sess}/{sess}_rec{rec}".format(dir=datadir, sbj=subject, sess=session, rec=rec)
        searchtoken = filetype
        
    filenames = os.listdir(searchdir)
    fn_found = []
    for fn in filenames:
        if searchtoken in fn:
            fn_found.append("{0}/{1}".format(searchdir, fn))
    if len(fn_found) == 0:
        raise IOError("Files of type '{0}' not found.".format(filetype))
    
    return fn_found

