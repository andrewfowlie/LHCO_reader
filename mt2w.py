#!/usr/bin/env python
"""
=========================
:math:`M_{T2}^W` variable
=========================

Functions for calculating :math:`M_{T2}^W` variable from
:literal:`MT2W-1.00a.zip` from
`Jiayin Gu's code <https://sites.google.com/a/ucdavis.edu/mass/>`_.

.. warning::
    You must compile Jiayin Gu's code in a library: \
    :literal:`gcc -shared -o libmt2w.so -fPIC mt2w_bisect.cpp`

.. warning::
    You must export the relevant bash variable to Jiayin Gu's code:\
    :literal:`export MT2W=/YOUR/PATH/TO//MT2W-1.00a

"""

###############################################################################

from __future__ import print_function
from __future__ import division

import os
import inspect

from scipy.weave import inline

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
    MT2W_PATH = os.environ["MT2W"]
except:
    raise ImportError("Couldn't read $MT2W bash variable")

HEADER = os.path.join(MT2W_PATH, "mt2w_bisect.h")
assert os.path.isfile(HEADER), "Cannot find header file: %s" % HEADER
LIBRARY = os.path.join(MT2W_PATH, "libmt2w.so")
assert os.path.isfile(LIBRARY), "Cannot find library: %s" % LIBRARY

###############################################################################

HEADERS = ['"mt2w_bisect.h"']
LIBRARY_DIRS = [MT2W_PATH]
LIBRARIES = ["mt2w"]

###############################################################################

def MT2W_wrapper(pl, pb1, pb2, pmiss, upper_bound=1000., error_value=0.):
    """
    Calculate :math:`M_{T2}^W` variable from
    :literal:`MT2W-1.00a.zip` from
    `Jiayin Gu's code <https://sites.google.com/a/ucdavis.edu/mass/>`_.

    :param pl: Lepton momentum
    :type pl: numpy.array
    :param pb1: Bottom-quark momentum, on lepton's side of event
    :type pb1: numpy.array
    :param pb2: Second bottom-quark momentum
    :type pb2: numpy.array
    :param pmiss: Missing momentum in event
    :type pmiss: numpy.array
    :param upper_bound: See :math:`M_{T2}^W` documentation
    :type upper_bound: float
    :param error_value: Value to be returned if no solutions or an error
    :type error_value: float

    :returns: :math:`M_{T2}^W`
    :rtype: float

    :Example:

    >>> import numpy as np

    >>> pl = np.array([123.92, 33.5126, 28.4367, 115.864])
    >>> pb1 = np.array([633.838, -4.35958, -18.0347, 633.54])
    >>> pb2 = np.array([53.8194, 46.0444, -0.864168, 27.4517])
    >>> pmiss = np.array([0., -75.1975, -9.53793, 0.])

    >>> MT2W = MT2W_wrapper(pl, pb1, pb2, pmiss)
    >>> print(MT2W)
    140.234222412

    >>> pl = np.array([265.594, 84.8964, 131.638, 214.485])
    >>> pb1 = np.array([67.6987, -59.7058, 13.5768, 28.6956])
    >>> pb2 = np.array([72.892, 114.745, 56.5649, 114.816])
    >>> pmiss = np.array([0., 60.9, -194.732, 0.])

    >>> MT2W = MT2W_wrapper(pl, pb1, pb2, pmiss)
    >>> print(MT2W)
    0.0
    """

    # Code for calculating MT2 variable in C++. This snippet returns
    # the MT2 variable.
    code = """
           using namespace std;
           mt2w_bisect::mt2w mt2w_event(upper_bound, error_value);
           mt2w_event.set_momenta(pl, pb1, pb2, pmiss);
	       const float mt2w = mt2w_event.get_mt2w();
           return_val = mt2w;
           """

    # Pass all Python arguments to C++
    frame = inspect.currentframe()
    args = inspect.getargvalues(frame)[0]

    # Compile and exectute code. If code is unchanged, it won't be recompiled
    lib_MT2W = inline(code,
                      args,
                      headers=HEADERS,
                      include_dirs=LIBRARY_DIRS,
                      library_dirs=LIBRARY_DIRS,
                      runtime_library_dirs=LIBRARY_DIRS,
                      libraries=LIBRARIES
                      )

    return lib_MT2W

def MT2W(lepton, bottom_1, bottom_2, MET):
    """
    Find :math:`M_{T2}` variable as defined in
    `arXiv:1411.4312 <http://arxiv.org/abs/1411.4312>`_ from
    four-momentum objects by calling the wrapper function.

    :param olepton: Lepton
    :type lepton: :class:`Object`
    :param bottom_1: First bottom, on lepton's side of event
    :type bottom_1: :class:`Object`
    :param bottom_2: Second bottom
    :type bottom_2: :class:`Object`
    :param MET: Invisible MET
    :type MET: :class:`Object`

    :returns: :math:`M_{T2}^W`
    :rtype: float

    :Example:

    >>> from kinetic_test import object_fourvector
    >>> from LHCO_reader import Fourvector
    >>> make_object = lambda vector: object_fourvector(Fourvector(vector))

    >>> pl = make_object([123.92, 33.5126, 28.4367, 115.864])
    >>> pb1 = make_object([633.838, -4.35958, -18.0347, 633.54])
    >>> pb2 = make_object([53.8194, 46.0444, -0.864168, 27.4517])
    >>> pmiss = make_object([0., -75.1975, -9.53793, 0.])

    >>> print(MT2W(pl, pb1, pb2, pmiss))
    140.235092163
    """
    # Make argument list for MTW2 function, converting four-vectors to
    # numpy arrays by slicing
    args = [oo.vector()[:] for oo in [lepton, bottom_1, bottom_2, MET]]

    # Call wrapper function
    return MT2W_wrapper(*args)

###############################################################################

if __name__ == "__main__":
    import doctest
    doctest.testmod()
