#!/usr/bin/env python
r"""
============
Kinetic test
============

Test kinematic variables against Oxbridge library.

>>> import oxbridge_kinetics

===================
Test :math:`M_{T2}`
===================

Check :literal:`helloWorld_Mt2_Without_Minuit2_Example.cpp`. Compile and
execute :literal:`helloWorld_Mt2_Without_Minuit2_Example.cpp` and check against
below results.

Functions to construct objects from
:literal:`helloWorld_Mt2_Without_Minuit2_Example.cpp`
inputs.

>>> E = lambda m, vec_p: (sum([p**2 for p in vec_p]) + m**2)**0.5
>>> p = lambda m, vec_p: LHCO_reader.Fourvector([E(m, vec_p)] + vec_p)
>>> object = lambda m, vec_p: object_fourvector(p(m, vec_p))

Inputs from  :literal:`helloWorld_Mt2_Without_Minuit2_Example.cpp`.

>>> vec_a = [410., 20., 0.]
>>> m_a = 100.
>>> vec_b = [-210., -300., 0.]
>>> m_b = 150.
>>> vec_MET = [-200., 280., 0.]
>>> m_MET = 100.

Build :class:`Object` for each object.

>>> object_a = object(m_a, vec_a)
>>> object_b = object(m_b, vec_b)
>>> object_MET = object(m_MET, vec_MET)

Calculate :math:`M_{T2}`.

>>> MT2 = oxbridge_kinetics.MT2(object_a, object_b, object_MET)
>>> print(MT2)
412.628838109

=====================
Test :math:`\alpha_T`
=====================

Compile and execute :literal:`AlphaT_Multijet_Calculator_Example.cpp` and
check against below results.

Ten lots of Oxbridge's example event
====================================

>>> j_1 = LHCO_reader.Fourvector([422.966, 410., 20., -20.])
>>> j_2 = LHCO_reader.Fourvector([398.166, -210., -300., 44.])
>>> event = event_fourvector([j_1, j_2] * 10)
>>> event.alpha_T()
0.5577197143955096

Event with no jets
==================

>>> event = LHCO_reader.Event()
>>> try:
...     event.alpha_T()
... except AssertionError as error:
...     print(error.message)
Calculating alpha_T for event with one or no jets

Event with one jet
==================

>>> j_1 = LHCO_reader.Fourvector([5., 1., 0., 0.])
>>> event = event_fourvector([j_1])
>>> try:
...     event.alpha_T()
... except AssertionError as error:
...     print(error.message)
Calculating alpha_T for event with one or no jets

Event with perfectly balanced jets
==================================

>>> j_1 = LHCO_reader.Fourvector([1., 1., 0., 0.])
>>> j_2 = LHCO_reader.Fourvector([1., -1., 0., 0.])
>>> event = event_fourvector([j_1, j_2])
>>> event.alpha_T()
0.5

Imbalanced back-to-back QCD events
==================================

>>> j_1 = LHCO_reader.Fourvector([1., 1., 0., 0.])
>>> j_2 = LHCO_reader.Fourvector([0.2, -0.2, 0., 0.])
>>> event = event_fourvector([j_1, j_2])
>>> event.alpha_T()
0.22360679774997896
>>> j_1 = LHCO_reader.Fourvector([1., -1., 0., 0.])
>>> j_2 = LHCO_reader.Fourvector([0.2, 0.2, 0., 0.])
>>> event = event_fourvector([j_1, j_2])
>>> event.alpha_T()
0.22360679774997896

Pascal Nef's first challenge
============================

Input data from :literal:`helloWorld_Mt2_Without_Minuit2_Example.cpp`.

>>> jet_data = [[5., 3., 4., 0.],
...             [13., 5., 12., 0.],
...             [25., 7., 24., 0.],
...             [17., 8., 15., 0.],
...             [41., 9., 40., 0.],
...             [61., 11., 60., 0.],
...             [37., 12., 35., 0.],
...             [85., 13., 84, 0.]
...             ]

>>> jets = []
>>> for jet in jet_data:
...     jets.append(LHCO_reader.Fourvector(jet))
>>> event = event_fourvector(jets)

Check :math:`\alpha_T`.

>>> event.alpha_T()
4.5602659028397481

Pascal Nef's second challenge
=============================

Input data from :literal:`helloWorld_Mt2_Without_Minuit2_Example.cpp`.

>>> jet_data = [[7., 3., 4., 0.],
...             [15., 5., 12., 0.],
...             [27., 7., 24., 0.],
...             [19., 8., 15., 0.],
...             [43., 9., 40., 0.],
...             [63., 11., 60., 0.],
...             [39., 12., 35., 0.],
...             [87., 13., 84, 0.]
...             ]

>>> jets = []
>>> for jet in jet_data:
...     jets.append(LHCO_reader.Fourvector(jet))
>>> event = event_fourvector(jets)

Check :math:`\alpha_T`.

>>> event.alpha_T()
4.5602659028397481

==========================================
Speed test for :math:`\alpha_T` algorithms
==========================================

Make data

>>> jet_data = [[7., 3., 4., 0.],
...             [15., 5., 12., 0.],
...             [27., 7., 24., 0.],
...             [19., 8., 15., 0.],
...             [43., 9., 40., 0.],
...             [63., 11., 60., 0.],
...             [39., 12., 35., 0.],
...             [187., 13., 84., 0.],
...             [117., 22., 12., 0.],
...             [182., 21., 14., 0.],
...             [125., 32., 12., 0.],
...             [123., 34., 13., 0.],
...             [181., 34., 25., 0.],
...             [148., 24., 19., 0.],
...             [119., 12., 32., 0.],
...             ]

>>> jets = []
>>> for jet in jet_data:
...     jets.append(LHCO_reader.Fourvector(jet))
>>> event = event_fourvector(jets)

Time calculations

>>> from time import time
>>> for algorithm in ["brute", "KK", "greedy", "CKK"]:
...     LHCO_reader.ALPHA_T_ALGORITHM = algorithm
...     t0 = time()
...     alpha_T = event.alpha_T()
...     delta_t = time() - t0
...     print(algorithm)
...     print(r"\alpha_T = %s" % alpha_T)
...     print("time = %s" % delta_t)
brute
\alpha_T = 1.27640720482
time = 0.031200170517
KK
\alpha_T = 1.27029936157
time = 0.00058388710022
greedy
\alpha_T = 1.25718917547
time = 0.000498056411743
CKK
\alpha_T = 1.27640720482
time = 0.00618004798889

===================
Test Razor variable
===================

>>> event = event_fourvector(jets)
>>> MET = object_fourvector(LHCO_reader.Fourvector([100., 10., 1., 9.]))
>>> event["MET"].append(MET)
>>> for algorithm in ["non_standard_brute", "non_standard_greedy"]:
...     LHCO_reader.RAZOR_ALGORITHM = algorithm
...     t0 = time()
...     R = event.razor_R()
...     delta_t = time() - t0
...     print(algorithm)
...     print(r"R = %s" % R)
...     print("time = %s" % delta_t)
non_standard_brute
R = 0.0631241278697
time = 1.9290368557
non_standard_greedy
R = 0.0609811416566
time = 0.00334310531616
"""

###############################################################################

from __future__ import print_function
from __future__ import division

import LHCO_reader
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


def object_fourvector(momentum, name="jet"):
    """
    Build a :class:`Object`, e.g. a jet, from a :class:`Fourvector` object.

    We must build a dictionary with keys:

    - :literal:`event`
    - :literal:`type`
    - :literal:`eta`
    - :literal:`phi`
    - :literal:`PT`
    - :literal:`jmass`
    - :literal:`ntrk`
    - :literal:`btag`
    - :literal:`hadem`

    .. warning::
        We don't set all keys from four-momenta - properties not determined \
        by four-momenta have arbitrary values.

    :param momentum: Fourvector of an object
    :type momentum: class:`Fourvector`
    :param names: Name of object
    :type names: string

    :returns: An object
    :rtype: :class:`Object`
    """

    # Find invariant mass of four-vector, catching errors related to
    # negative masses
    try:
        mass = abs(momentum)
    except AssertionError as error:
        warnings.warn(error.message)
        mass = 0.

    # Don't need all of this data - for many things, just put something
    # arbitrary
    dictionary = {"event": 1,
                  "type": 1,
                  "eta": momentum.eta(),
                  "phi": momentum.phi(),
                  "PT": momentum.PT(),
                  "jmass": mass,
                  "ntrk": 1,
                  "btag": 0,
                  "hadem": 1.
                  }

    return LHCO_reader.Object(dictionary=dictionary, name=name)


def event_fourvector(momenta, names=None):
    """
    Build an :class:`Event` from a set of :class:`Fourvector` objects.

    :param momenta: List of fourmomenta of objects in an event
    :type momenta: list of class:`Fourvector`
    :param names: Names of objects
    :type names: list of strings

    :returns: An event
    :rtype: :class:`Event`
    """

    # Empty event
    event = LHCO_reader.Event()

    # Make names of objects, if necessary. Pick jets by default
    if names is None:
        names = ["jet"] * len(momenta)

    # Append all objects
    for name, momentum in zip(names, momenta):
        object_ = object_fourvector(momentum, name=name)
        event[name].append(object_)

    return event

###############################################################################

if __name__ == "__main__":
    import doctest
    doctest.testmod()
