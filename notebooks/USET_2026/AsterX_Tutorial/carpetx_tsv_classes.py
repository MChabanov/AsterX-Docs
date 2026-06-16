import pathlib
import re

import numpy as np


class OneDimTsv_OneOut:
    def __init__(self, path):
        self.path = path
        self.output = pathlib.Path(self.path)

        # all 1D line-output files in this output folder
        self.onedim_files = [str(p) for p in self.output.rglob("*.it*.tsv")]
        assert self.onedim_files, "No 1D data found in this output folder!"

        # unique variable names, parsed from the filenames
        self.vars = []
        for elem in self.onedim_files:
            var = elem.split('.')[-4].rpartition('/')[2]
            if var not in self.vars:
                self.vars.append(var)

        # which iterations exist per variable and direction
        self.iterations = {var: {'x': [], 'y': [], 'z': []} for var in self.vars}
        for sss in self.onedim_files:
            sss_list = sss.split('.')
            direc = sss_list[-2]
            it    = int(sss_list[-3].partition('t')[2])
            var   = sss_list[-4].rpartition('/')[2]
            self.iterations[var][direc].append(it)

        for var in self.iterations:
            for direc in self.iterations[var]:
                self.iterations[var][direc].sort()

    def get_vars(self):
        return self.vars

    def get_iterations(self, var, direc):
        return np.array(self.iterations[var][direc])

    def get_path(self):
        return self.path

    def get_path_list(self):
        return self.onedim_files


class NormsTsv_OneOut:
    def __init__(self, path):
        self.path = path
        self.output = pathlib.Path(self.path)

        # norm (time-series) files: tsv files without an `.itNNNN` tag
        self.norm_files = []
        for p in self.output.rglob("*.tsv"):
            string = str(p)
            if not re.search(r"[.]it\d+", string):
                self.norm_files.append(string)
        assert self.norm_files, "No norm data found in this output folder!"

        # unique variable names, parsed from the filenames
        self.vars = []
        for elem in self.norm_files:
            var = elem.split('.')[-2].rpartition('/')[2]
            if var not in self.vars:
                self.vars.append(var)

    def get_vars(self):
        return self.vars

    def get_path_list(self):
        return self.norm_files

    def get_path(self):
        return self.path


class LoadTsv:
    def __init__(self, path):
        self.path = path

        # load data; atleast_2d keeps a single-row file as a (1, ncols) array
        self.data = np.atleast_2d(np.loadtxt(self.path))

        # read the header line to build the column lookup
        with open(self.path) as f:
            first_line = f.readline()

        # strip the Cactus `thorn::varname` (or `:varname`) prefixes
        self.strings_use = []
        for elem in first_line.split():
            if elem == '#':
                continue
            string = elem.partition('::')[2]
            self.strings_use.append(string if string else elem.partition(':')[2])

        self.look_up = {name: i for i, name in enumerate(self.strings_use)}

    # public

    def get(self, sss):
        return self.data[:, self.look_up[sss]]

    def get_keys(self):
        return self.strings_use

    def get_path(self):
        return self.path


class LoadAllTimersTsv:
    def __init__(self, path):
        self.path = path

        # it=0 row and it>0 rows carry different column sets
        self.data_0 = np.loadtxt(self.path, skiprows=2, max_rows=1, ndmin=2)
        self.data_1 = np.loadtxt(self.path, skiprows=4, ndmin=2)

        # read the four header lines (third one is not needed)
        with open(self.path) as f:
            first_line  = f.readline()
            second_line = f.readline()
            f.readline()
            fourth_line = f.readline()

        strings_first  = re.split(r"\t+", first_line)
        strings_second = re.split(r"\t+", second_line)
        strings_fourth = re.split(r"\t+", fourth_line)

        self.which_clock_val = strings_first[0].split()[-1]
        self.clock_units_val = strings_first[1].split()[-1]

        # column names for the it=0 row and the it>0 rows
        self.strings_use_0 = list(strings_second)
        self.strings_use_1 = list(strings_fourth)
        self.strings_use_0[-1] = self.strings_use_0[-1].partition("\n")[0]
        self.strings_use_1[-1] = self.strings_use_1[-1].partition("\n")[0]

        # we base our methods on the fact
        # that strings_use_0 is a subset of
        # strings_use_1
        assert set(self.strings_use_0).issubset(self.strings_use_1), \
            "it=0 column values are not a subset of the column values for it>0"

        self.look_up_0 = {name: i for i, name in enumerate(self.strings_use_0)}
        self.look_up_1 = {name: i for i, name in enumerate(self.strings_use_1)}

    def __get_0(self, sss):
        return self.data_0[:, self.look_up_0[sss]]

    def __get_keys_0(self):
        return self.strings_use_0

    def __get_1(self, sss):
        return self.data_1[:, self.look_up_1[sss]]

    def __get_keys_1(self):
        return self.strings_use_1

    # public

    def get_path(self):
        return self.path

    def get_clock(self):
        return self.which_clock_val

    def get_clock_units(self):
        return self.clock_units_val

    def get(self, sss):
        if sss in self.strings_use_0:
            return np.append(self.data_0[0, self.look_up_0[sss]], self.data_1[:, self.look_up_0[sss]])
        else:
            return self.data_1[:, self.look_up_1[sss]]

    def get_keys(self):
        return self.__get_keys_1()


def LoadNormsTsv(OneOut, var):
    path_list = OneOut.get_path_list()

    this_string = None
    for elem in path_list:
        if re.search(r"/" + re.escape(var) + r"\.tsv$", elem):
            this_string = elem
            break

    assert this_string is not None, "Requested tsv file not found!"

    if var == 'AllTimers':
        return LoadAllTimersTsv(this_string)
    else:
        return LoadTsv(this_string)


def LoadOneDimTsv(OneOut, var, direction, it):
    path_list = OneOut.get_path_list()

    this_string = None
    for elem in path_list:
        if re.search(r"/" + re.escape(var) + r"\.it" + str(it).zfill(6) + r"\." + re.escape(direction) + r"\.tsv$", elem):
            this_string = elem
            break

    assert this_string is not None, "Requested tsv file not found!"

    return LoadTsv(this_string)
