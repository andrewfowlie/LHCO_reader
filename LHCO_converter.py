#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

"""

Convert LHCO and ROOT files. You must

export DELPHES=YOUR/PATH/

"""

__author__ = "Andrew Fowlie"
__copyright__ = "Copyright 2015"
__credits__ = ["Luca Marzola"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Andrew Fowlie"
__email__ = "andrew.fowlie@monash.edu.au"
__status__ = "Production"

###############################################################################

import subprocess
import os
import warnings

###############################################################################

try:
    delphes_path = os.environ["DELPHES"]
    lhco2root = os.path.join(delphes_path, "lhco2root")
    root2lhco = os.path.join(delphes_path, "root2lhco")
    assert os.path.isfile(lhco2root)
    assert os.path.isfile(root2lhco)
except:
    raise Exception("Couldn't read $DELPHES bash variable for root2lhco and lhco2root")

###############################################################################


def unique_name(f_name):
    """
    Find a file name that does not exist based on a supplied file name.

    Arguments:
    f_name -- File name

    Returns:
    trial_name -- Unique file name, based on f_name
    """
    base = os.path.splitext(f_name)[0]
    extension = os.path.splitext(f_name)[1]

    trial_name = f_name
    ii = 0
    while os.path.isfile(trial_name):
        ii += 1
        trial_name = base + "_" + str(ii) + extension

    if trial_name != f_name:
        warnings.warn("Requested %s changed to %s" % (f_name, trial_name))

    return trial_name

###############################################################################


def LHCO_ROOT(LHCO_name, ROOT_name=None):
    """
    Convert a file from LHCO to ROOT. Files won't be
    overwritten - instead a new unique file-name is chosen.

    Arguments:
    LHCO_name -- Name of LHCO file to be read
    ROOT_name -- Name of ROOT file to be written

    Returns:
    ROOT_name -- Name of ROOT file written
    """
    # Make non-existent ROOT name
    if not ROOT_name:
        ROOT_name = os.path.splitext(LHCO_name)[0] + ".root"
    ROOT_name = unique_name(ROOT_name)

    if not os.path.isfile(LHCO_name):
        raise Exception("Could not find LHCO file: %s" % LHCO_name)

    # Convert LHCO to ROOT
    command = [lhco2root, ROOT_name, LHCO_name]
    process = subprocess.Popen(command, stdout=open(os.devnull, 'w'), stderr=subprocess.PIPE)
    error = process.communicate()[1]

    if "Error" in error or not os.path.isfile(ROOT_name):
        raise Exception("Could not convert to ROOT")

    return ROOT_name

###############################################################################


def ROOT_LHCO(ROOT_name, LHCO_name=None):
    """
    Convert a file from ROOT to LHCO. Files won't be
    overwritten - instead a new unique file-name is chosen.

    Arguments:
    ROOT_name -- Name of ROOT file to be read
    LHCO_name -- Name of LHCO file to be written

    Returns:
    LHCO_name -- Name of LHCO file written
    """
    # Make non-existent LHCO name
    if not LHCO_name:
        LHCO_name = os.path.splitext(ROOT_name)[0] + ".lhco"
    LHCO_name = unique_name(LHCO_name)

    if not os.path.isfile(root2lhco):
        raise Exception("Could not find %s" % root2lhco)

    if not os.path.isfile(ROOT_name):
        raise Exception("Could not find ROOT file %s" % ROOT_name)

    # Convert ROOT to LHCO
    command = [root2lhco, ROOT_name, LHCO_name]
    process = subprocess.Popen(command, stdout=open(os.devnull, 'w'), stderr=subprocess.PIPE)
    error = process.communicate()[1]

    if "Error" in error or not os.path.isfile(LHCO_name):
        raise Exception("Could not convert to LHCO")

    return LHCO_name
