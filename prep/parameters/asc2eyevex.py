if __name__ == "__main__":
    import os.path
    execfile("{}/../{}".format(*os.path.split(__file__)))
else:
    from inspect import currentframe
    from os.path import basename, splitext
    importing_file = currentframe().f_back.f_code.co_filename
    if splitext(basename(importing_file))[0] != splitext(basename(__file__))[0]:
        raise ValueError("Attempted to import inappropriate parameters; names of the code file and the parameter file don't match")


datadir = "/users/ito/datasets/osaka/RAWDATA/Human"
prepdir = "/users/ito/datasets/osaka/PREPROCESSED/Human"

dataset_info = (
    # (sbj, sess, rec, blk, tasktype, taskID)
    ("MK", "", 1, 1, "Free", 3, ),
    ("MK", "", 2, 1, "Free", 3, ),
    ("MK", "", 3, 1, "Free", 3, ),
    ("MK", "", 4, 1, "Free", 3, ),

    ("SI", "", 1, 1, "Free", 3, ),
    ("SI", "", 2, 1, "Free", 3, ),
    ("SI", "", 3, 1, "Free", 3, ),
    ("SI", "", 4, 1, "Free", 3, ),

    ("SO", "", 1, 1, "Free", 3, ),
    ("SO", "", 2, 1, "Free", 3, ),
    ("SO", "", 3, 1, "Free", 3, ),
    ("SO", "", 4, 1, "Free", 3, ),
)

pxlperdeg = 29.69
