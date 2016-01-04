#!/usr/bin/env python
"""
=================
Oxbridge kinetics
=================

Link with your `Oxbridge kinetics library
<www.hep.phy.cam.ac.uk/~lester/mt2>`_.

.. warning::
    You must `export OXBRIDGE=YOUR/PATH/` to your \
    build of oxbridge_kinetics and `export OXBRIDGE_LIB=YOUR/PATH/`
    to the directory containing your oxbridge_kinetics library,
    :literal:`liboxbridgekinetics-1.0.so.1`.

This module is intended to be be imported by :mod:`LHCO_reader`.

==========
References
==========

If you use this interface, please cite:
* `C.G.Lester and D.J.Summers <http://arxiv.org/abs/hep-ph/9906349>`_
* `A.J.Barr, C.G.Lester and P.Stephens <http://arxiv.org/abs/hep-ph/0304226>`_
* `C.G.Lester and  A.J.Barr <http://arxiv.org/abs/0708.1028>`_
* `H.Cheng and Z.Han <http://arxiv.org/abs/0810.5178>`_
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
    OXBRIDGE_PATH = os.environ["OXBRIDGE"]
    OXBRIDGE_LIB_PATH = os.environ["OXBRIDGE_LIB"]
except:
    raise ImportError("Couldn't read $OXBRIDGE "
                      "or ${OXBRIDGE_LIB} bash variable")

HEADER = os.path.join(OXBRIDGE_PATH, "Mt2/ChengHanBisect_Mt2_332_Calculator.h")
assert os.path.isfile(HEADER), "Cannot find header file: %s" % HEADER
LIBRARY = os.path.join(OXBRIDGE_LIB_PATH, "liboxbridgekinetics-1.0.so.1")
assert os.path.isfile(LIBRARY), "Cannot find library: %s" % LIBRARY

###############################################################################

# Libraries and directories for compiling oxbridge kinetic code
HEADERS = ['"Mt2/ChengHanBisect_Mt2_332_Calculator.h"']
LIBRARY_DIRS = [OXBRIDGE_LIB_PATH]
LIBRARIES = ["oxbridgekinetics-1.0"]
INCLUDE_DIRS = [OXBRIDGE_PATH]

###############################################################################


def MT2_wrapper(m_1, p1_x, p1_y, m_2, p2_x, p2_y, m_MET, MET_x, MET_y):
    """
    Find :math:`M_{T2}` variable as defined in
    `arXiv:1411.4312 <http://arxiv.org/abs/1411.4312>`_ by compiling
    code from the Oxbridge kinetics library on the fly.

    :param m_1: Mass of first object
    :type m_1: float
    :param p1_x: x-momentum of first object
    :type p1_x: float
    :param p1_y: y-momentum of first object
    :type p1_y: float
    :param m_2: Mass of second object
    :type m_2: float
    :param p2_x: x-momentum of second object
    :type p2_x: float
    :param p2_y: y-momentum of second object
    :type p2_y: float
    :param m_MET: Mass of MET object
    :type m_MET: float
    :param MET_x: x-momentum of MET object
    :type MET_x: float
    :param MET_y: y-momentum of MET object
    :type MET_y: float

    :returns: :math:`M_{T2}` variable
    :rtype: float

    :Example:

    >>> p1_x, p1_y = 410., 20.
    >>> p2_x, p2_y = -210., -300.
    >>> MET_x, MET_y = -200., 280.
    >>> m_1 = 100.
    >>> m_2 = 150.
    >>> m_MET = 100.
    >>> MT2 = MT2_wrapper(m_1, p1_x, p1_y, m_2, p2_x, p2_y, m_MET, MET_x, MET_y)
    >>> print(MT2)
    412.628838109
    """

    # Code for calculating MT2 variable in C++. This snippet returns
    # the MT2 variable.
    code = """
           std::cout.setstate(std::ios_base::failbit); // Suppress cout
           Mt2::ChengHanBisect_Mt2_332_Calculator mt2Calculator;
           Mt2::LorentzTransverseVector vis_A(Mt2::TwoVector(p1_x, p1_y), m_1);
           Mt2::LorentzTransverseVector vis_B(Mt2::TwoVector(p2_x, p2_y), m_2);
           Mt2::TwoVector pT_Miss(MET_x, MET_y);
           const double mt2 = mt2Calculator.mt2_332(vis_A, vis_B, pT_Miss, m_MET);
           return_val = mt2;
           """

    # Pass all Python arguments to C++
    frame = inspect.currentframe()
    args = inspect.getargvalues(frame)[0]

    # Compile and execute code. If code is unchanged, it won't be recompiled
    lib_MT2 = inline(code,
                     args,
                     headers=HEADERS,
                     include_dirs=INCLUDE_DIRS,
                     library_dirs=LIBRARY_DIRS,
                     runtime_library_dirs=LIBRARY_DIRS,
                     libraries=LIBRARIES
                     )

    return lib_MT2


def MT2(object_1, object_2, MET):
    """
    Find :math:`M_{T2}` variable as defined in
    `arXiv:1411.4312 <http://arxiv.org/abs/1411.4312>`_ from
    four-momentum objects by calling the wrapper function.

    :param object_1: First visible object
    :type object_1: :class:`Object`
    :param object_2: Second visible object
    :type object_2: :class:`Object`
    :param MET: Invisible MET
    :type MET: :class:`Object`

    :returns: :math:`M_{T2}` variable
    :rtype: float
    """
    # Make argument list for MT2 function
    args = []
    for object_ in [object_1, object_2, MET]:
        vector = object_.vector()
        args += [abs(vector), vector[1], vector[2]]

    # Call wrapper function
    return MT2_wrapper(*args)

###############################################################################

if __name__ == "__main__":
    import doctest
    doctest.testmod()
