#!/usr/bin/env python
"""
==============
LHCO_converter
==============

Convert `LHCO <http://madgraph.phys.ucl.ac.be/Manual/lhco.html>`_ and ROOT
files.

.. warning:
    You must `export DELPHES=YOUR/PATH/` to your \
    `Delphes <https://cp3.irmp.ucl.ac.be/projects/delphes>`_ path.

This module is intended to be be imported by :mod:`LHCO_reader`.
"""

###############################################################################

from __future__ import print_function
from __future__ import division

import subprocess
import os
import warnings

###############################################################################

__author__ = "Andrew Fowlie"
__copyright__ = "Copyright 2015"
__credits__ = ["Luca Marzola"]
__license__ = "GPL"
__maintainer__ = "Andrew Fowlie"
__email__ = "Andrew.Fowlie@Monash.Edu.Au"
__status__ = "Production"

###############################################################################

try:
    DELPHES_PATH = os.environ["DELPHES"]
    LHCO_TO_ROOT = os.path.join(DELPHES_PATH, "lhco2root")
    ROOT_TO_LHCO = os.path.join(DELPHES_PATH, "root2lhco")
    assert os.path.isfile(LHCO_TO_ROOT)
    assert os.path.isfile(ROOT_TO_LHCO)
except:
    raise ImportError("Couldn't read $DELPHES bash variable")

###############################################################################


def unique_name(f_name):
    """
    Find a file name that does not exist based on a supplied file name::

        unique_name("what_if_this_exists.txt")

    :param f_name: File name
    :type f_name: string

    :returns: Unique file name, based on f_name
    :rtype: string
    """
    base = os.path.splitext(f_name)[0]
    extension = os.path.splitext(f_name)[1]

    trial_name = f_name
    trial_number = 0
    while os.path.isfile(trial_name):
        trial_number += 1
        trial_name = base + "_" + str(trial_number) + extension

    if trial_name != f_name:
        warnings.warn("Requested %s changed to %s" % (f_name, trial_name))

    return trial_name

###############################################################################


def LHCO_ROOT(LHCO_name, ROOT_name=None):
    """
    Convert a file from LHCO to ROOT::

        LHCO_ROOT("my_LHCO_file.lhco", "my_new_ROOT_file.root")

    If no ROOT_name is supplied, a name based on LHCO_name is chosen.

    .. warning::
        By default, files *should not* be overwritten -
        instead a new unique file-name is chosen.

    :param LHCO_name: Name of LHCO file to be read
    :type LHCO_name: string
    :param ROOT_name: Name of ROOT file to be written
    :type ROOT_name: string

    :returns: Name of ROOT file written
    :rtype: string
    """
    # Make non-existent ROOT name
    if not ROOT_name:
        ROOT_name = os.path.splitext(LHCO_name)[0] + ".root"
    ROOT_name = unique_name(ROOT_name)

    if not os.path.isfile(LHCO_name):
        raise IOError("Could not find LHCO file: %s" % LHCO_name)

    # Convert LHCO to ROOT
    command = [LHCO_TO_ROOT, ROOT_name, LHCO_name]
    process = subprocess.Popen(command,
                               stdout=open(os.devnull, 'w'),
                               stderr=subprocess.PIPE
                               )
    error = process.communicate()[1]

    if "Error" in error or not os.path.isfile(ROOT_name):
        raise RuntimeError("Could not convert to ROOT")

    return ROOT_name

###############################################################################


def ROOT_LHCO(ROOT_name, LHCO_name=None):
    """
    Convert a file from ROOT to LHCO::

        ROOT_LHCO("my_ROOT_file.root", "my_new_LHCO_file.lhco")

    If no ROOT_name is supplied, a name based on LHCO_name is chosen.

    .. warning::
        By default, files *should not* be overwritten -
        instead a new unique file-name is chosen.

    :param ROOT_name: Name of ROOT file to be read
    :type ROOT_name: string
    :param LHCO_name: Name of LHCO file to be written
    :type LHCO_name: string

    :returns: Name of LHCO file written
    :rtype: string
    """
    # Make non-existent LHCO name
    if not LHCO_name:
        LHCO_name = os.path.splitext(ROOT_name)[0] + ".lhco"
    LHCO_name = unique_name(LHCO_name)

    if not os.path.isfile(ROOT_TO_LHCO):
        raise IOError("Could not find %s" % ROOT_TO_LHCO)

    if not os.path.isfile(ROOT_name):
        raise IOError("Could not find ROOT file %s" % ROOT_name)

    # Convert ROOT to LHCO
    command = [ROOT_TO_LHCO, ROOT_name, LHCO_name]
    process = subprocess.Popen(command,
                               stdout=open(os.devnull, 'w'),
                               stderr=subprocess.PIPE
                               )
    error = process.communicate()[1]

    if "Error" in error or not os.path.isfile(LHCO_name):
        raise RuntimeError("Could not convert to LHCO")

    return LHCO_name
