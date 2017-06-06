import os
import re
import datetime
import time
from collections import defaultdict

import numpy as np
import scipy
import scipy.stats
import scipy.spatial.distance
import sklearn.cluster

import odml


def convert_dataset_to_odml_info(dataset, channel_names, units, dtypes):
    sbj, sess, rec, blk, site, idx_granular_channel, granular_quality, idxs_bad_channel = dataset

    sectname = "Dataset/LFPData"
    section_info = {
        "Dataset": {"name": "Dataset", "type": "dataset", "subsections": ["LFPData",]},
        "Dataset/LFPData": {"name": "SpikeData", "type": "dataset/neural_data", "subsections": []},
    }
    props = {sectname: []}
    props[sectname].append({"name": "NumChannels", "value": len(channel_names), "unit": None, "dtype": dtypes["NumChannels"]})
    props[sectname].append({"name": "ChannelNames", "value": channel_names, "unit": None, "dtype": None})
    if len(idxs_bad_channel) > 0:
        props[sectname].append({"name": "NumBadChannels", "value": len(idxs_bad_channel), "unit": None, "dtype": dtypes["NumBadChannels"]})
        props[sectname].append({"name": "BadChannels", "value": [channel_names[i] for i in idxs_bad_channel], "unit": None, "dtype": None})
    else:
        props[sectname].append({"name": "NumBadChannels", "value": 0, "unit": None, "dtype": dtypes["NumBadChannels"]})
    props[sectname].append({"name": "GranularLayerIdentificationQuality", "value": granular_quality, "unit": None, "dtype": None})
    if idx_granular_channel is not None:
        props[sectname].append({"name": "GranularLayerChannel", "value": channel_names[idx_granular_channel], "unit": None, "dtype": None})

    return section_info, props


class odMLFactory(object):
    def __init__(self, section_info={}, default_props={}, filename='', strict=True):
        self.sect_info = section_info
        self.def_props = default_props
        self.strict = strict
        if filename:
            self._sections = self.__get_sections_from_file(filename)
        else:
            self._sections = {}
            for sectname in self.__get_top_section_names():
                self._sections[sectname] = self.__gen_section(sectname)

    def __get_sections_from_file(self, filename):
        # load odML from file
        with open(filename, 'r') as fd_odML:
            metadata = odml.tools.xmlparser.XMLReader().fromFile(fd_odML)
        sections = {}
        for sect in metadata.sections:
            sections[sect.name] = sect
        return sections

    def __get_top_section_names(self):
        topsectnames = []
        for key in self.sect_info:
            if '/' not in key:
                topsectnames.append(key)
        return topsectnames

    def __add_property(self, sect, prop, strict=True):
        if sect.contains(odml.Property(prop['name'], None)):
            sect.remove(sect.properties[prop['name']])
        elif strict is True:
            raise ValueError("Property '{0}' does not exist in section '{1}'.".format(prop['name'], sect.name))
        name = prop['name']
        if isinstance(prop['value'], list):
            value = prop['value']
        else:
            value = odml.Value(data=prop['value'], unit=prop['unit'], dtype=prop['dtype'])
        sect.append(odml.Property(name, value))

    def __gen_section(self, name, parent=''):
        longname = parent + name
        sect = odml.Section(name=name, type=self.sect_info[longname]['type'])

        # add properties
        if longname in self.def_props:
            for prop in self.def_props[longname]:
                self.__add_property(sect, prop, strict=False)

        # add subsections
        if 'subsections' in self.sect_info[longname]:
            for subsectname in self.sect_info[longname]['subsections']:
                sect.append(self.__gen_section(subsectname, longname+'/'))

        return sect

    def __get_section_from_longname(self, sectname):
        def get_subsect(sect, names):
            if len(names) == 0:
                return sect
            else:
                return get_subsect(sect.sections[names[0]], names[1:])

        names = sectname.split('/')
        if names[0] not in self._sections:
            return None
        else:
            return get_subsect(self._sections[names[0]], names[1:])

    def put_values(self, properties):
        for sectname, sectprops in properties.items():
            sect = self.__get_section_from_longname(sectname)
            if sect is None:
                raise ValueError("Invalid section name '{0}'".format(sectname))
            else:
                for prop in sectprops:
                    self.__add_property(sect, prop, strict=False)

    def get_odml(self, author, version=None):
        metadata = odml.Document(author, datetime.date.today(), version)
        for sect in self._sections.values():
            metadata.append(sect)
        return metadata

    def save_odml(self, filename, author, version=None):
        metadata = self.get_odml(author, version)
        odml.tools.xmlparser.XMLWriter(metadata).write_file(filename)


def print_metadata(metadata):
    def print_section(sect, ntab=0, tabstr='    '):
        tabs = tabstr * ntab
        print("{0}{1} (type: {2})".format(tabs, sect.name, sect.type))
        tabs = tabstr * (ntab + 1)
        for prop in sect.properties:
            if isinstance(prop.value, list):
                data = [str(x.data) for x in prop.value]
                if prop.value == []:
                    unit = ""
                    dtype = ""
                elif prop.value[0].unit is None:
                    unit = ""
                    dtype = prop.value[0].dtype
                else:
                    unit = prop.value[0].unit
                    dtype = prop.value[0].dtype
                print("{0}{1}: [{2}] {3} (dtype: {4})".format(tabs, prop.name, ', '.join(data), unit, dtype))
            else:
                unit = "" if prop.value.unit is None else prop.value.unit
                print("{0}{1}: {2} {3} (dtype: {4})".format(tabs, prop.name, prop.value.data, unit, prop.value.dtype))
        print

        for subsect in sect.sections:
            print_section(subsect, ntab+1)

    print("Version {0}, Created by {1} on {2}".format(metadata.version, metadata.author, metadata.date))
    print
    for sect in metadata.sections:
        print_section(sect)
        print


if __name__ == "__main__":
    # import parameters from the configuration file
    from parameters.gen_lfp_catalog_odml import *

    for dataset in datasets:
        sbj, sess, rec, blk, site, _, _, _ = dataset
        dataset_name = "{}_rec{}_blk{}_{}".format(sess, rec, blk, site)
        print "\n{sbj}:{dataset_name} ".format(**locals())

        # construct odML structure and save it in a file
        section_info, props = convert_dataset_to_odml_info(dataset, channel_names, odml_units, odml_dtypes)
        odml_factory = odMLFactory(section_info, strict=False)
        odml_factory.put_values(props)
        filename_odml = "{}/{}_LFP.odml".format(savedir, dataset_name)
        odml_factory.save_odml(filename_odml, odml_author, odml_version)
        print "\tLFP metadata saved in {0}\n".format(filename_odml)

        # # print out the odML structure for a check
        # print_metadata(odml_factory.get_odml(odml_author, odml_version))

        print "\tProcessing of {} done\n".format(dataset_name)
