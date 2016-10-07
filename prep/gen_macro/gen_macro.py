import os
from argparse import ArgumentParser

import numpy as np


labels_ch = [
    '00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
    '0a', '0b', '0c', '0d', '0e', '0f', '0g', '0h', '0i', '0j',
    '0k', '0l', '0m', '0n'
    ]


# parse command line options
parser = ArgumentParser()
parser.add_argument("--dataname", default="20010101rec1blk1")
parser.add_argument("--datadir", default="data")
parser.add_argument("--headerdir", default="d_info")
parser.add_argument("--macronames", default=["clust017.macroSpkSortingPlexon24Clust", "clust017.macroSpkSortingPlexon24SpkDet"])
parser.add_argument("--macrodir", default="MACRO")
arg = parser.parse_args()

# compute data length
filename = "{0}/{1}/{1}.b00".format(arg.datadir, arg.dataname)
with open(filename, 'rb') as f:
    f.seek(0, 2)
    length = f.tell() / 4
# length = 10
width = length

# compute STD of the given data
stds = []
for label_ch in labels_ch:
    filename = "{0}/{1}/{1}.b{2}".format(arg.datadir, arg.dataname, label_ch)
    data = np.fromfile(filename, dtype='>f4')
    stds.append(data.std())
std = np.mean(stds)
# std = 1.0

noise_level_clustering_3SD = 3.0 * std * 1000
noise_level = 4.5 * std * 1000

distdir = "{}".format(arg.macrodir)
for macroname in arg.macronames:
    infile = "{}/{}.template".format(arg.macrodir, macroname)
    outfile = "{}/{}".format(distdir, macroname)
    with open(infile, "r") as f_in, open(outfile, "w") as f_out:
        template_text = f_in.read()
        f_out.write(template_text.format(**locals()))
        # print template_text.format(**locals())

distdir = "{}/{}".format(arg.headerdir, arg.dataname)
os.mkdir(distdir)
infile = "{}/d_info.template".format(arg.headerdir)
with open(infile, "r") as f_in:
    template_text = f_in.read()
    # print template_text.format(**locals())
    for label_ch in labels_ch:
        outfile = "{0}/{1}.b{2}".format(distdir, arg.dataname, label_ch)
        with open(outfile, "w") as f_out:
            f_out.write(template_text.format(**locals()))
