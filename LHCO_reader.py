#!/usr/bin/env python
"""
============
Introduction
============

Read an `LHCO <http://madgraph.phys.ucl.ac.be/Manual/lhco.html>`_
(or ROOT) file, from e.g. 
`PGS <http://www.physics.ucdavis.edu/~conway/research/software/pgs/pgs4-general.html>`_ 
or `Delphes <https://cp3.irmp.ucl.ac.be/projects/delphes>`_ into
convenient Python classes.

The code is object-oriented. A LHCO file is parsed into several objects:

- :class:`Events` (inherits the list class): List of events
- :class:`Event` (inherits the dictionary class): Dictionary of objects in an
  event, e.g. :literal:`electron`
- :class:`Objects` (inherits the list class): List of objects of a particular
  type in an event, e.g. list of electrons
- :class:`Object` (inherits the dictionary class): Dictionary of properties of
  an object, e.g. an electron's transverse momentum

============
Simple usage
============

Simple usage is e.g::

    events = Events("example.lhco")  # Load "example.lhco" LHCO file
    print(events[11]["electron"][0]["PT"])  # Print transverse momentum

that code loads an LHCO file, and prints the transverse momentum of the first
electron in the eleventh event.

===============
Dictionary keys
===============

The :class:`Event` dictionary's keys are

- :literal:`electron`
- :literal:`muon`
- :literal:`tau`
- :literal:`jet`
- :literal:`MET` (missing transverse energy)
- :literal:`photon`

The :class:`Object` dictionary's keys from the LHCO file are

- :literal:`event`
- :literal:`type`
- :literal:`eta`
- :literal:`phi`
- :literal:`PT`
- :literal:`jmass`
- :literal:`ntrk`
- :literal:`btag`
- :literal:`hadem`

event and type are integers, and other properties are floats.

We add various additional properties, including a function :func:`vector()`,
which returns a four-momentum object.

Complicated kinematic variables could be included from the oxbridge kinetics
library.

>>> object_1 = events[0]["jet"][0]
>>> object_2 = events[0]["jet"][1]
>>> MET = events[0]["MET"][0]
>>> from oxbridge_kinetics import MT2
>>> MT2(object_1, object_2, MET)
53.305931964300186
"""
###############################################################################

from __future__ import print_function
from __future__ import division

import os
import warnings
import inspect

import numpy as np
import partition_problem as pp

from math import pi
from numpy import cos, sin
from collections import OrderedDict
from prettytable import PrettyTable as pt

###############################################################################

__author__ = "Andrew Fowlie"
__copyright__ = "Copyright 2015"
__credits__ = ["Luca Marzola"]
__license__ = "GPL"
__maintainer__ = "Andrew Fowlie"
__email__ = "Andrew.Fowlie@Monash.Edu.Au"
__status__ = "Production"

###############################################################################

# Constants for my classes

###############################################################################

# Tuple of object properties in LHCO file, these are row headings in
# LHCO format, see http://madgraph.phys.ucl.ac.be/Manual/lhco.html
PROPERTIES = ("event",
              "type",
              "eta",
              "phi",
              "PT",
              "jmass",
              "ntrk",
              "btag",
              "hadem",
              "dummy1",
              "dummy2"
              )

# Tuple of properties suitable for printing
PRINT_PROPERTIES = ("eta",
                    "phi",
                    "PT",
                    "jmass",
                    "ntrk",
                    "btag",
                    "hadem"
                    )

# Dictionary of object names that appear in event, index corresponds
# to the number in the LHCO convention. Dictionary rather than list,
# because number 5 is missing.
NUMBERS = (0, 1, 2, 3, 4, 6)
NAMES = ("photon",
         "electron",
         "muon",
         "tau",
         "jet",
         "MET"
         )
NAMES_DICT = dict(zip(NUMBERS, NAMES))
HEADINGS = ["Object"] + list(PRINT_PROPERTIES)
EMPTY_DICT = dict.fromkeys(NAMES)

LEPTON = ["electron", "muon", "tau"]

# Default algorithms for partition problems
ALPHA_T_ALGORITHM = "CKK"
RAZOR_ALGORITHM = "non_standard_brute"

###############################################################################

# Classes for storing LHCO file data

###############################################################################


class Events(list):
    """
    Contains all events in an LHCO file as a list of :class:`Event` objects.

    Includes functions to parse an LHCO file into a list of :class:`Event`
    objects.

    Inherits the list class; it is itself a list with an integer index. Each
    entry in the list is an :class:`Event`. Simple usage e.g.

    :Example:

    >>> events=Events("example.lhco")
    >>> print(events)
    +------------------+--------------+
    | Number of events | 10000        |
    | Description      | example.lhco |
    +------------------+--------------+

    Each list entry is itself an :class:`Event` class - a class designed to
    store a single event.

    :Example:

    >>> print(events[100])
    +----------+-------+-------+-------+-------+------+------+-------+
    |  Object  |  eta  |  phi  |   PT  | jmass | ntrk | btag | hadem |
    +----------+-------+-------+-------+-------+------+------+-------+
    |  photon  |  0.13 | 2.109 | 16.68 |  0.0  | 0.0  | 0.0  |  0.05 |
    |  photon  | 0.217 | 6.149 |  5.7  |  0.0  | 0.0  | 0.0  |  0.08 |
    | electron | -2.14 | 1.816 | 16.76 |  0.0  | -1.0 | 0.0  |  0.01 |
    | electron | 1.183 | 4.001 | 15.84 |  0.0  | 1.0  | 0.0  |  0.0  |
    |   jet    | 0.434 | 6.161 |  17.5 |  4.12 | 4.0  | 0.0  |  0.57 |
    |   jet    | 1.011 | 0.196 | 10.71 |  1.63 | 4.0  | 0.0  |  3.64 |
    |   jet    | 1.409 | 4.841 |  5.11 |  0.53 | 2.0  | 0.0  |  0.3  |
    |   MET    |  0.0  | 4.221 | 16.18 |  0.0  | 0.0  | 0.0  |  0.0  |
    +----------+-------+-------+-------+-------+------+------+-------+
    >>> print(events[100]["electron"].order("phi"))
    +----------+-------+-------+-------+-------+------+------+-------+
    |  Object  |  eta  |  phi  |   PT  | jmass | ntrk | btag | hadem |
    +----------+-------+-------+-------+-------+------+------+-------+
    | electron | 1.183 | 4.001 | 15.84 |  0.0  | 1.0  | 0.0  |  0.0  |
    | electron | -2.14 | 1.816 | 16.76 |  0.0  | -1.0 | 0.0  |  0.01 |
    +----------+-------+-------+-------+-------+------+------+-------+
    >>> print(events[100]["electron"][0])
    +----------+-------+-------+-------+-------+------+------+-------+
    |  Object  |  eta  |  phi  |   PT  | jmass | ntrk | btag | hadem |
    +----------+-------+-------+-------+-------+------+------+-------+
    | electron | 1.183 | 4.001 | 15.84 |  0.0  | 1.0  | 0.0  |  0.0  |
    +----------+-------+-------+-------+-------+------+------+-------+
    >>> print(events[100]["electron"][0]["PT"])
    15.84

    >>> events=Events("example.lhco", n_events=250)
    >>> print(events)
    +------------------+--------------+
    | Number of events | 250          |
    | Description      | example.lhco |
    +------------------+--------------+
    """

    def __init__(self,
                 f_name=None,
                 list_=None,
                 cut_list=None,
                 n_events=None,
                 description=None
                 ):
        """
        Parse an LHCO or ROOT file into a list of :class:`Event` objects.

        It is possible to initialise an :class:`Events` class without a LHCO
        or a ROOT file, and later append events to the list.

        .. warning::
            Parsing a ROOT file requires
            (and attempts to import) :mod:`LHCO_converter`.

        :param f_name: Name of an LHCO or ROOT file, including path
        :type f_name: str
        :param list_: A list for initialising events
        :type list_: list
        :param cut_list: Cuts applied to events and their acceptance
        :type cut_list: list
        :param n_events: Number of events to read from LHCO file
        :type n_events: int
        :param description: Information about events
        :type description: string
        """

        # Ordinary initialisation
        if list_:
            super(self.__class__, self).__init__(list_)

        # Save number of events read
        self.n_events = n_events

        # Save cuts
        self.cut_list = cut_list
        if not self.cut_list:
            self.cut_list = []

        # Save description
        self.description = description
        if not self.description and f_name:
            self.description = f_name

        # Find/make and parse LHCO file
        self.LHCO_name = None
        if not f_name:
            warnings.warn("Events class without a LHCO or ROOT file")
        elif not os.path.isfile(f_name):  # Check that file exists
            raise IOError("File does not exist: %s" % f_name)
        else:
            # Consider file-type - ROOT or LHCO
            f_extension = os.path.splitext(f_name)[1]
            if f_extension == ".root":
                # Convert ROOT to LHCO file
                import LHCO_converter
                self.LHCO_name = LHCO_converter.ROOT_LHCO(f_name)
            elif f_extension == ".lhco":
                self.LHCO_name = f_name
            else:
                raise IOError("Unknown file extension: %s" % f_name)

            self.__parse()  # Parse file

            if not len(self):  # Check whether any events were parsed
                warnings.warn("No events were parsed")
            elif not self.__exp_number():  # Check number of parsed events
                warnings.warn("Couldn't read total number of events from LHCO")
            elif not self.__exp_number() == len(self):
                warnings.warn("Did not parse all events in file")

    def add_event(self, event):
        """
        Add an :class:`Event` object built from LHCO data to
        the :class:`Events` class, i.e. append the :class:`Events` list with
        a new :class:`Event` object.

        :param event: List of lines in LHCO file for event
        :type event: :class:`Event`
        """
        self.append(Event(event))

    def __parse(self):
        """
        Parse an LHCO file into individual :class:`Event` objects.

        Our strategy is to loop over file, line by line, and split the
        file into individual events. A new event begins with a :literal:`0`
        and ends an existing event.

        To read about the LHCO format, see
        `here <http://madgraph.phys.ucl.ac.be/Manual/lhco.html>`_

        This attribute is intended to be private - i.e. only called from within
        this class itself.
        """

        def parse_lines(file_):
            """
            Generator that ignores comments and empty lines.

            :param file_: Name of file
            :type file_: file

            :returns: Lines that should be parsed
            :rtype: iterator
            """
            for line in file_:
                line = line.strip()
                if line and not line.startswith("#"):
                    yield line

        n_events = self.n_events
        event = None

        with open(self.LHCO_name, 'r') as LHCO_file:

            for line in parse_lines(LHCO_file):

                if line.startswith("0"):  # New event in file

                    if event:  # Parse previous event, if there is one
                        self.add_event(event)  # Add Event class
                    event = [line]  # New event - reset event list

                    # Don't parse more than a particular number of events
                    if n_events and len(self) == n_events:
                        warnings.warn("Didn't parse all LHCO events, by request")
                        return
                else:

                    # If there is not a "0", line belongs to current event,
                    # not a new event - add it to event
                    try:
                        event.append(line)
                    except AttributeError:
                        warnings.warn("Possibly an event did not start with 0")
                        event = [line]

            # Parse final event in file - because there isn't a following event
            # the final event won't be parsed as above
            self.add_event(event)

    def number(self, anti_lepton=False):
        """
        Count the total numbers of objects in all events, e.g. total number
        of electrons, jets, muons etc.

        Store the information in a dictionary.

        :param anti_lepton: Whether anti-leptons treated separately
        :type anti_lepton: bool

        :returns: Dictionary of numbers of each object, indexed by e.g. \
        :literal:`electron`
        :rtype: dict

        :Example:

        >>> print(events.number())
        +--------+----------+------+------+-------+-------+---------------+--------------+
        | photon | electron | muon | tau  |  jet  |  MET  | Total objects | Total events |
        +--------+----------+------+------+-------+-------+---------------+--------------+
        |  1350  |  17900   |  3   | 1512 | 33473 | 10000 |     64238     |    10000     |
        +--------+----------+------+------+-------+-------+---------------+--------------+
        >>> print(events.number(anti_lepton=True))
        +--------+----------+------+-----+-------+-------+---------------+-----------+----------+---------------+--------------+
        | photon | electron | muon | tau |  jet  |  MET  | anti-electron | anti-muon | anti-tau | Total objects | Total events |
        +--------+----------+------+-----+-------+-------+---------------+-----------+----------+---------------+--------------+
        |  1350  |   8976   |  2   | 732 | 33473 | 10000 |      8924     |     1     |   780    |     64238     |    10000     |
        +--------+----------+------+-----+-------+-------+---------------+-----------+----------+---------------+--------------+
        """

        # Initialise dictionary for numbers of objects
        number = PrintDict()

        number_names = list(NAMES)
        if anti_lepton:
            for name in LEPTON:
                anti_name = "anti-" + name
                number_names.append(anti_name)

        for name in number_names:
            number[name] = 0

        # Find numbers of objects by looping through events
        for event in self:
            for name in number_names:
                number[name] += event.number(anti_lepton=anti_lepton)[name]

        number["Total objects"] = sum(number.values())
        number["Total events"] = len(self)

        return number

    def __exp_number(self):
        """
        Returns expected number of events in LHCO file.

        Number of events written in the LHCO file, as
        :literal:`##  Number of Event :` or as
        :literal:`# | Number of events |`.

        This attribute is intended to be private - i.e. only called from within
        this class itself.

        :returns: Number of events expected in LHCO file
        :rtype: int
        """

        # Read expected number of events from LHCO file
        exp_events = None

        try:
            with open(self.LHCO_name, 'r') as LHCO_file:
                for line in LHCO_file:
                    if line.strip().startswith("##  Number of Event"):
                        exp_events = int(line.split(":")[1])
                        break
                    if line.strip().startswith("# | Number of events |"):
                        exp_events = int(line.split("|")[2])
                        break
        except IOError:
            warnings.warn("Couldn't read expected number of events")

        return exp_events

    def __add__(self, other):
        """
        Combine :class:`Event` classes.

        "Add" two :class:`Event` classes together, making a new :class:`Event`
        class with all the events from each :class:`Event` class.

        :param other: An class:`Event` class of events
        :type other: class:`Event`

        :returns: Combined class:`Event` class of events
        :rtype: class:`Event`

        :Example:

        >>> events + events
        +------------------+-----------------------------+
        | Number of events | 20000                       |
        | Description      | example.lhco + example.lhco |
        +------------------+-----------------------------+
        """

        if not isinstance(other, Events):
            raise ValueError("Can only add Events with other Events object")

        # Make a description
        try:
            description = self.description + " + " + other.description
        except TypeError:
            warnings.warn("Couldn't find descriptions")
            description = None

        # New set of events in Events class
        combined = Events(list_=list.__add__(self, other),
                          description=description
                          )

        return combined

    def acceptance(self):
        """
        :returns: Combined acceptance of all cuts
        :rtype: float

        >>> events = Events("example.lhco")
        >>> tau = lambda event: event.number()["tau"] == 1
        >>> events.cut(tau)
        >>> events.acceptance()
        0.8656
        """
        combined_acceptance = 1.
        for (_, acceptance) in self.cut_list:
            combined_acceptance *= acceptance
        return combined_acceptance

    def number_original(self):
        """
        Reconstruct number of events in original set of events from cut
        efficiencies.

        .. warning::
            For this to be correct, you must always remove events \
            using :funct:`cut_events`.

        :returns: Original number of events
        :rtype: integer

        >>> events = Events("example.lhco")
        >>> tau = lambda event: event.number()["tau"] == 1
        >>> events.cut(tau)
        >>> events.number_original()
        10000
        """
        return int(len(self) / self.acceptance())

    def error_acceptance(self):
        r"""
        Statistical error of estimate of acceptance (from :func:`acceptance`)
        found from binomial statistics.

        .. math::
            \sigma = \sqrt{\frac{p (1 - p)}{n}}

        :returns: Statistical error of acceptance
        :rtype: float

        >>> events = Events("example.lhco")
        >>> tau = lambda event: event.number()["tau"] == 1
        >>> events.cut(tau)
        >>> events.error_acceptance()
        0.0034108157382069172
        """
        acceptance = self.acceptance()
        number_original = self.number_original()
        error = (acceptance * (1. - acceptance) / number_original)**0.5
        return error

    def summarize_cuts(self):
        """
        Make a summary table of cuts applied to events
        from :literal:`cut_list`.

        :returns: Table of cuts
        :rtype: string

        >>> events = Events("example.lhco")
        >>> tau = lambda event: event.number()["tau"] == 1
        >>> events.cut(tau)
        >>> print(events.summarize_cuts())
        +------------------------------------------------+------------------+
        | tau = lambda event: event.number()["tau"] == 1 | 0.8656           |
        | Combined acceptance                            | 0.8656           |
        | Error acceptance                               | 0.00341081573821 |
        +------------------------------------------------+------------------+
        """
        table = pt(header=False)

        for cut_number, (cut, acceptance) in enumerate(self.cut_list):

            # Inspect source code
            try:
                cut_string = inspect.getsource(cut).strip()
            except IOError:
                warnings.warn("Did not inspect source. Probably running interactively")
                cut_string = "Cut %s" % cut_number

            table.add_row([cut_string, str(acceptance)])

        # Add information about combined acceptance
        if self.cut_list:
            table.add_row(["Combined acceptance", str(self.acceptance())])
            table.add_row(["Error acceptance", str(self.error_acceptance())])

        table.align = "l"
        return table.get_string()

    def __str__(self):
        """
        Make a string of an :class:`Events` class.

        Rather than attempting to return thousands of events, return summary
        information about set of events.

        :returns: String of :class:`Events` class
        :rtype: string

        :Example:

        >>> print(events)
        +------------------+--------------+
        | Number of events | 10000        |
        | Description      | example.lhco |
        +------------------+--------------+
        """

        table = pt(header=False)
        table.add_row(["Number of events", len(self)])
        table.add_row(["Description", self.description])
        table.align = "l"
        summary = table.get_string()

        # Add information about cuts
        if self.cut_list:
            summary += "\n\n" + self.summarize_cuts()

        return summary

    def __repr__(self):
        """
        Represent object as a brief description. Representation of such a
        big list is anyway unreadable. See :func:`__str__`.

        :returns: String of :class:`Events` class
        :rtype: string
        """
        return self.__str__()

    def cut(self, cut):
        """
        Apply a cut, i.e. remove events that fail a test.

        The cut should be a function that takes event as an argument,
        and returns `True` or `False`. If `True`, the event is removed.

        :param cut: Cut to apply to events
        :type cut: function

        :Example:

        >>> events = Events("example.lhco")
        >>> tau = lambda event: event.number()["tau"] == 1
        >>> events.cut(tau)
        >>> print(events)
        +------------------+--------------+
        | Number of events | 8656         |
        | Description      | example.lhco |
        +------------------+--------------+
        <BLANKLINE>
        +------------------------------------------------+------------------+
        | tau = lambda event: event.number()["tau"] == 1 | 0.8656           |
        | Combined acceptance                            | 0.8656           |
        | Error acceptance                               | 0.00341081573821 |
        +------------------------------------------------+------------------+
        """

        len_org = len(self)  # Remember original length
        if not len_org:
            raise ValueError("No events")

        # Loop list in reverse order, removing events if they fail the cut.
        # Must be reverse order, because otherwise indices are
        # shifted when events are removed from the list.
        iterator = reversed(list(enumerate(self)))

        for event_number, event in iterator:
            if cut(event):
                del self[event_number]  # NB del is much faster than remove

        len_new = len(self)
        if not len_new:
            warnings.warn("All events were cut")

        # Append information about the cut to events
        acceptance = len_new / len_org
        self.cut_list.append([cut, acceptance])

    def cut_objects(self, name, cut):
        """
        Apply a cut to a set of objects inside an event, for all events.

        E.g. cut all jets with :math:`P_T < 30`.

        :param name: Name of object, e.g. :literal:`electron`
        :type name: string
        :param cut: Cut to apply to objects
        :type cut: function

        :Example:
        >>> events = Events("example.lhco")
        >>> print(events[10])
        +----------+--------+-------+-------+-------+------+------+-------+
        |  Object  |  eta   |  phi  |   PT  | jmass | ntrk | btag | hadem |
        +----------+--------+-------+-------+-------+------+------+-------+
        | electron | -1.359 | 1.056 | 79.51 |  0.0  | -1.0 | 0.0  |  0.02 |
        | electron | -0.327 |  2.84 | 26.27 |  0.0  | 1.0  | 0.0  |  0.01 |
        |   tau    | -0.182 | 3.595 |  65.3 |  0.0  | -1.0 | 0.0  |  3.84 |
        |   jet    | 1.449  | 5.194 | 61.81 |  6.15 | 8.0  | 0.0  |  1.47 |
        |   jet    | 1.736  | 0.393 |  8.72 |  1.53 | 7.0  | 0.0  |  1.86 |
        |   MET    |  0.0   |  1.15 |  8.66 |  0.0  | 0.0  | 0.0  |  0.0  |
        +----------+--------+-------+-------+-------+------+------+-------+
        >>> PT = lambda object_: object_["PT"] < 30.
        >>> events.cut_objects("jet", PT)
        >>> print(events[10])
        +----------+--------+-------+-------+-------+------+------+-------+
        |  Object  |  eta   |  phi  |   PT  | jmass | ntrk | btag | hadem |
        +----------+--------+-------+-------+-------+------+------+-------+
        | electron | -1.359 | 1.056 | 79.51 |  0.0  | -1.0 | 0.0  |  0.02 |
        | electron | -0.327 |  2.84 | 26.27 |  0.0  | 1.0  | 0.0  |  0.01 |
        |   tau    | -0.182 | 3.595 |  65.3 |  0.0  | -1.0 | 0.0  |  3.84 |
        |   jet    | 1.449  | 5.194 | 61.81 |  6.15 | 8.0  | 0.0  |  1.47 |
        |   MET    |  0.0   |  1.15 |  8.66 |  0.0  | 0.0  | 0.0  |  0.0  |
        +----------+--------+-------+-------+-------+------+------+-------+
        >>> print(events)
        +------------------+--------------+
        | Number of events | 10000        |
        | Description      | example.lhco |
        +------------------+--------------+
        <BLANKLINE>
        +------------------------------------------+-----+
        | PT = lambda object_: object_["PT"] < 30. | 1.0 |
        | Combined acceptance                      | 1.0 |
        | Error acceptance                         | 0.0 |
        +------------------------------------------+-----+
        """

        if name not in NAMES:
            warnings.warn("Name not recognised: %s" % name)
            return

        for event in self:
            event[name].cut_objects(cut)

        # Note that objects were cut with 100% efficiency
        self.cut_list.append([cut, 1.])

    def column(self, name, prop):
        """
        Make a list of all e.g. electron's transverse momentum, :math:`P_T`::

            events.column("electron", "PT")

        :param name: Name of object, e.g. :literal:`electron`
        :type name: string
        :param prop: Property of object, e.g. :literal:`PT`, transverse \
        momentum
        :type prop: string

        :returns: List of all e.g. electron's :math:`P_T`
        :rtype: list
        """

        if name not in NAMES:
            warnings.warn("Name not recognised: %s" % name)
            return

        if prop not in PROPERTIES:
            warnings.warn("Property not recognised: %s" % prop)
            return

        # Make a list of the desired property
        column = []
        for event in self:
            column += [objects[prop] for objects in event[name]]

        return column

    def mean(self, name, prop):
        """
        Find mean of e.g. electron's transverse momentum.

        :param name: Name of object, e.g. :literal:`electron`
        :type name: string
        :param prop: Property of object, e.g. :literal:`PT`, transverse \
        momentum
        :type prop: string

        :returns: Mean of e.g. electron's :math:`P_T`
        :rtype: float

        :Example:

        >>> events.mean("electron", "PT")
        123.39502178770951
        """

        if name not in NAMES:
            warnings.warn("Name not recognised: %s" % name)
            return

        if prop not in PROPERTIES:
            warnings.warn("Property not recognised: %s" % prop)
            return

        mean = np.mean(self.column(name, prop))

        return mean

    def plot(self, name, prop):
        """
        Show a 1-dimensional histogram of an object's property.

        Plot basic histogram with matplotlib with crude titles and axis
        labels. The histogram is normalised to one.

        :param name: Name of object, e.g. :literal:`electron`
        :type name: string
        :param prop: Property of object, e.g. :literal:`PT`, transverse \
        momentum
        :type prop: string

        :Example:

        >>> events.plot("electron", "PT")

        .. figure::  plot.png
            :align:   center
        """
        import matplotlib.pyplot as plt

        if name not in NAMES:
            warnings.warn("Name not recognised: %s" % name)
            return

        if prop not in PROPERTIES:
            warnings.warn("Property not recognised: %s" % prop)
            return

        data = self.column(name, prop)  # Make data into a column

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(data, 50, normed=1, facecolor='Crimson', alpha=0.9)
        ax.grid()  # Add grid lines

        ax.set_title(name)  # Titles etc
        ax.set_xlabel(prop)
        ax.set_ylabel("Frequency")
        plt.rc('font', size=20)  # Fonts - LaTeX possible, but slow

        plt.show()  # Show the plot on screen

    def __getslice__(self, i, j):
        """
        Slicing an :class:`Events` class  returns another :class:`Events` class
        rather than a list.

        :param i: Opening index
        :type i: integer
        :param j: Closing index
        :type j: integer

        :returns: Slice of :class:`Events` class
        :rtype: :class:`Events`

        :Example:

        >>> print(events[:100])
        +------------------+--------------------------------+
        | Number of events | 100                            |
        | Description      | Events 0 to 99 in example.lhco |
        +------------------+--------------------------------+
        """
        events = Events(list_=list.__getslice__(self, i, j))
        events.description = "Events {} to {} in {}".format(i, j-1, self.description)
        return events

    def __mul__(self, other):
        """
        Multiplying an :class:`Events` class returns another :class:`Events`
        class rather than a list.

        :param other: Number to multiply by
        :type other: int

        :returns: Numerous copies of events in :class:`Events` class
        :rtype: :class:`Events`

        :Example:

        >>> print(events * 5)
        +------------------+--------------------------+
        | Number of events | 50000                    |
        | Description      | 5 copies of example.lhco |
        +------------------+--------------------------+
        """
        events = Events(list_=list.__mul__(self, other))
        events.description = "{} copies of {}".format(other, self.description)
        return events

    def __rmul__(self, other):
        """ See :func:`__mul__`. """
        return self.__mul__(other)

    def LHCO(self, LHCO_name, over_write=False):
        """
        Write events in LHCO format.

        .. warning::
            By default, files *should not* be overwritten.

        :param LHCO_name: Name of LHCO file to be written
        :type LHCO_name: string
        :param over_write: Whether to over-write an existing file
        :type over_write: bool

        :returns: Name of LHCO file written
        :rtype: string

        :Example:

        >>> f_name = events[:10].LHCO("events.lhco", over_write=True)
        >>> same_events = Events(f_name)
        >>> print(events[1])
        +----------+--------+-------+-------+-------+------+------+-------+
        |  Object  |  eta   |  phi  |   PT  | jmass | ntrk | btag | hadem |
        +----------+--------+-------+-------+-------+------+------+-------+
        | electron | -0.95  | 0.521 |  41.2 |  0.0  | 1.0  | 0.0  |  0.0  |
        | electron | -2.153 | 4.152 | 36.66 |  0.0  | -1.0 | 0.0  |  0.0  |
        |   jet    | -2.075 | 2.035 | 37.34 |  4.58 | 6.0  | 0.0  |  1.54 |
        |   jet    | 2.188  | 4.088 | 12.71 |  2.77 | 4.0  | 0.0  |  2.14 |
        |   MET    |  0.0   | 5.378 |  9.6  |  0.0  | 0.0  | 0.0  |  0.0  |
        +----------+--------+-------+-------+-------+------+------+-------+
        >>> print(same_events[1])
        +----------+--------+-------+-------+-------+------+------+-------+
        |  Object  |  eta   |  phi  |   PT  | jmass | ntrk | btag | hadem |
        +----------+--------+-------+-------+-------+------+------+-------+
        | electron | -0.95  | 0.521 |  41.2 |  0.0  | 1.0  | 0.0  |  0.0  |
        | electron | -2.153 | 4.152 | 36.66 |  0.0  | -1.0 | 0.0  |  0.0  |
        |   jet    | -2.075 | 2.035 | 37.34 |  4.58 | 6.0  | 0.0  |  1.54 |
        |   jet    | 2.188  | 4.088 | 12.71 |  2.77 | 4.0  | 0.0  |  2.14 |
        |   MET    |  0.0   | 5.378 |  9.6  |  0.0  | 0.0  | 0.0  |  0.0  |
        +----------+--------+-------+-------+-------+------+------+-------+
        """
        if os.path.isfile(LHCO_name) and not over_write:
            raise IOError("Cannot overwrite %s" % LHCO_name)

        preamble = """LHCO file created with LHCO_reader (https://github.com/innisfree/LHCO_reader).
See http://madgraph.phys.ucl.ac.be/Manual/lhco.html for a description of the LHCO format."""
        print(comment(preamble), file=open(LHCO_name, "w"), end="\n\n")
        print(comment(self), file=open(LHCO_name, "a"), end="\n\n")

        for event_number, event in enumerate(self):
            print("# Event number:", event_number, file=open(LHCO_name, "a"))
            event.LHCO(LHCO_name)

        return LHCO_name

    def ROOT(self, ROOT_name):
        """
        Write events in root format.

        .. warning::
            Files *should not* be overwritten instead a new \
            unique file name is chosen.

         .. warning::
            Requires (and attempts to import) :mod:`LHCO_converter`.

        :param ROOT_name: Name of LHCO file to be written
        :type ROOT_name: string

        :returns: Name of ROOT file written
        :rtype: string

        :Example:

        >>> ROOT_name = events.ROOT("ROOT_events.root")
        >>> ROOT_events = Events(ROOT_name)
        >>> print(events.number())
        +--------+----------+------+------+-------+-------+---------------+--------------+
        | photon | electron | muon | tau  |  jet  |  MET  | Total objects | Total events |
        +--------+----------+------+------+-------+-------+---------------+--------------+
        |  1350  |  17900   |  3   | 1512 | 33473 | 10000 |     64238     |    10000     |
        +--------+----------+------+------+-------+-------+---------------+--------------+
        >>> print(ROOT_events.number())
        +--------+----------+------+------+-------+-------+---------------+--------------+
        | photon | electron | muon | tau  |  jet  |  MET  | Total objects | Total events |
        +--------+----------+------+------+-------+-------+---------------+--------------+
        |  1350  |  17900   |  3   | 1512 | 33473 | 10000 |     64238     |    10000     |
        +--------+----------+------+------+-------+-------+---------------+--------------+
        """

        # Make an LHCO file
        LHCO_name = self.LHCO(".LHCO_reader.lhco", over_write=True)

        # Convert LHCO to ROOT
        import LHCO_converter
        ROOT_name = LHCO_converter.LHCO_ROOT(LHCO_name, ROOT_name)
        if not os.path.isfile(ROOT_name):
            raise RuntimeError("Could not convert to ROOT")

        return ROOT_name

###############################################################################


class PrintDict(OrderedDict):
    """
    An ordinary dictionary with a nice printing function.
    """
    def __str__(self):
        """
        Make an easy to read table of the dictionary contents.

        :returns: Table of the dictionary contents
        :rtype: string

        """

        # Make table of dictionary keys and entries
        table = pt(self.keys())
        table.add_row(self.values())

        return table.get_string()

###############################################################################


class Objects(list):
    """
    Objects in an LHCO event of a particular type.

    E.g., a list of all electron objects in an LHCO event. Each individual
    electron object is an :class:`Object` class.
    """

    def order(self, prop, reverse=True):
        """
        Order objects by a particular property, e.g. order all jets by
        transverse momentum :math:`P_T`, :literal:`PT`.

        By default, the objects are listed in *reverse* order, biggest to
        smallest. E.g., if sorted by :math:`P_T`, the hardest jet appears
        first.

        Sorting is in place *and* a sorted list is returned.

        :param prop: Property by which to sort objects, e.g. sort by \
        :literal:`PT`
        :type prop: string
        :param reverse: Order objects in reverse order
        :type reverse: bool

        :returns: List of objects, now ordered
        :rtype: :class:`Objects`
        """

        if not self[0].get(prop):
            warnings.warn("Property not recognised: %s" % prop)
            return

        # Simply sort the list by the required property
        self.sort(key=lambda obj: obj[prop], reverse=reverse)

        return self

    def __str__(self):
        """
        Return a nice readable table format.

        :returns: Table of the dictionary contents
        :rtype: string
        """

        # Make table of event
        table = pt(HEADINGS)

        # Add rows to the table
        for object_ in self:
            table.add_row(object_._row())

        return table.get_string()

    def __add__(self, other):
        """
        Add :class:`Objects` together, returning a new :class:`Objects` class.

        E.g. you might wish to add :literal:`electron` with :literal:`muon` to
        make an :class:`Objects` class of all leptons.

        :param other: Objects to combine
        :type other: class:`Objects`

        :returns: Combined class:`Objects` classes
        :rtype: class:`Objects`

        :Example:

        >>> e = events[100]
        >>> e["jet+electron"] = e["jet"] + e["electron"]
        >>> print(e["jet+electron"])
        +----------+-------+-------+-------+-------+------+------+-------+
        |  Object  |  eta  |  phi  |   PT  | jmass | ntrk | btag | hadem |
        +----------+-------+-------+-------+-------+------+------+-------+
        |   jet    | 0.434 | 6.161 |  17.5 |  4.12 | 4.0  | 0.0  |  0.57 |
        |   jet    | 1.011 | 0.196 | 10.71 |  1.63 | 4.0  | 0.0  |  3.64 |
        |   jet    | 1.409 | 4.841 |  5.11 |  0.53 | 2.0  | 0.0  |  0.3  |
        | electron | -2.14 | 1.816 | 16.76 |  0.0  | -1.0 | 0.0  |  0.01 |
        | electron | 1.183 | 4.001 | 15.84 |  0.0  | 1.0  | 0.0  |  0.0  |
        +----------+-------+-------+-------+-------+------+------+-------+
        >>> for ii, e in enumerate(events):
        ...     events[ii]["jet+electron"] = e["jet"] + e["electron"]
        >>> print(events[0]["jet+electron"])
        +----------+--------+-------+--------+-------+------+------+-------+
        |  Object  |  eta   |  phi  |   PT   | jmass | ntrk | btag | hadem |
        +----------+--------+-------+--------+-------+------+------+-------+
        |   jet    | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |   jet    | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        |   jet    | 0.811  | 6.028 | 17.49  |  3.47 | 8.0  | 0.0  |  2.37 |
        |   jet    | 0.596  | 0.853 | 12.47  |  2.53 | 7.0  | 0.0  |  1.26 |
        |   jet    | -1.816 | 0.032 |  6.11  |  1.18 | 0.0  | 0.0  |  0.56 |
        |   jet    | 0.508  | 1.421 |  6.01  |  0.94 | 7.0  | 0.0  |  2.59 |
        | electron | -0.745 | 4.253 | 286.72 |  0.0  | -1.0 | 0.0  |  0.0  |
        | electron | -0.073 | 4.681 | 44.56  |  0.0  | 1.0  | 0.0  |  0.0  |
        +----------+--------+-------+--------+-------+------+------+-------+
        >>> print(events[0]["jet+electron"].order("PT"))
        +----------+--------+-------+--------+-------+------+------+-------+
        |  Object  |  eta   |  phi  |   PT   | jmass | ntrk | btag | hadem |
        +----------+--------+-------+--------+-------+------+------+-------+
        | electron | -0.745 | 4.253 | 286.72 |  0.0  | -1.0 | 0.0  |  0.0  |
        |   jet    | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |   jet    | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        | electron | -0.073 | 4.681 | 44.56  |  0.0  | 1.0  | 0.0  |  0.0  |
        |   jet    | 0.811  | 6.028 | 17.49  |  3.47 | 8.0  | 0.0  |  2.37 |
        |   jet    | 0.596  | 0.853 | 12.47  |  2.53 | 7.0  | 0.0  |  1.26 |
        |   jet    | -1.816 | 0.032 |  6.11  |  1.18 | 0.0  | 0.0  |  0.56 |
        |   jet    | 0.508  | 1.421 |  6.01  |  0.94 | 7.0  | 0.0  |  2.59 |
        +----------+--------+-------+--------+-------+------+------+-------+
        >>> print(events.number())
        +--------+----------+------+------+-------+-------+---------------+--------------+
        | photon | electron | muon | tau  |  jet  |  MET  | Total objects | Total events |
        +--------+----------+------+------+-------+-------+---------------+--------------+
        |  1350  |  17900   |  3   | 1512 | 33473 | 10000 |     64238     |    10000     |
        +--------+----------+------+------+-------+-------+---------------+--------------+
        """
        combination = Objects(list.__add__(self, other))

        return combination

    def __getslice__(self, i, j):
        """
        Slicing returns another :class:`Objects` rather than a list.

        :param i: Opening index
        :type i: integer
        :param j: Closing index
        :type j: integer

        :returns: Sliced class:`Objects` class
        :rtype: class:`Objects`

        :Example:

        >>> print(events[0]["jet"][:2])
        +--------+--------+-------+--------+-------+------+------+-------+
        | Object |  eta   |  phi  |   PT   | jmass | ntrk | btag | hadem |
        +--------+--------+-------+--------+-------+------+------+-------+
        |  jet   | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |  jet   | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        +--------+--------+-------+--------+-------+------+------+-------+
        """
        return Objects(list.__getslice__(self, i, j))

    def __mul__(self, other):
        """
        Multiplying returns another :class:`Objects` class rather than a list.

        :param other: Number to multiply by
        :type other: int

        :returns: Numerous copies of objects in :class:`Objects` class
        :rtype: :class:`Objects`

        :Example:

        >>> print(events[0]["jet"][:2] * 5)
        +--------+--------+-------+--------+-------+------+------+-------+
        | Object |  eta   |  phi  |   PT   | jmass | ntrk | btag | hadem |
        +--------+--------+-------+--------+-------+------+------+-------+
        |  jet   | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |  jet   | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        |  jet   | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |  jet   | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        |  jet   | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |  jet   | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        |  jet   | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |  jet   | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        |  jet   | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |  jet   | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        +--------+--------+-------+--------+-------+------+------+-------+
        """
        return Objects(list.__mul__(self, other))

    def __rmul__(self, other):
        """ See :func:`__mul__`. """
        return self.__mul__(other)

    def LHCO(self, f_name):
        """
        Write objects in LHCO format.

        :param f_name:  Name of LHCO file to be written
        :type f_name: string
        """
        if self:
            self.order("PT")
            for object_ in self:
                object_.LHCO(f_name)

    def cut_objects(self, cut):
        """
        Apply a cut to a set of objects inside an event.

        E.g. cut all jets with :math:`P_T < 30`.

        :param cut: Cut to apply to objects
        :type cut: function

        :Example:
        >>> events = Events("example.lhco")
        >>> objects = events[0]["jet"]
        >>> print(objects)
        +--------+--------+-------+--------+-------+------+------+-------+
        | Object |  eta   |  phi  |   PT   | jmass | ntrk | btag | hadem |
        +--------+--------+-------+--------+-------+------+------+-------+
        |  jet   | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |  jet   | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        |  jet   | 0.811  | 6.028 | 17.49  |  3.47 | 8.0  | 0.0  |  2.37 |
        |  jet   | 0.596  | 0.853 | 12.47  |  2.53 | 7.0  | 0.0  |  1.26 |
        |  jet   | -1.816 | 0.032 |  6.11  |  1.18 | 0.0  | 0.0  |  0.56 |
        |  jet   | 0.508  | 1.421 |  6.01  |  0.94 | 7.0  | 0.0  |  2.59 |
        +--------+--------+-------+--------+-------+------+------+-------+
        >>> PT = lambda object_: object_["PT"] < 30.
        >>> objects.cut_objects(PT)
        >>> print(objects)
        +--------+--------+-------+--------+-------+------+------+-------+
        | Object |  eta   |  phi  |   PT   | jmass | ntrk | btag | hadem |
        +--------+--------+-------+--------+-------+------+------+-------+
        |  jet   | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |  jet   | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        +--------+--------+-------+--------+-------+------+------+-------+
        """
        # Loop list in reverse order, removing events if they fail the cut.
        # Must be reverse order, because otherwise indices are
        # shifted when events are removed from the list.
        iterator = reversed(list(enumerate(self)))

        for object_number, object_ in iterator:
            if cut(object_):
                del self[object_number]  # NB del is much faster than remove

    def pick_charge(self, charge):
        """
        Make a :class:`Objects` class of particles or anti-particles of same
        type. E.g. all anti-electrons or electrons in event.

        .. warning::
            Only applies to electron, mu and tau, as other objects either \
            have no measured charge or no anti-particles.

        :param charge: Charge of leptons required, e.g. :literal:`-1`
        :type lepton: integer

        :returns: Particles or anti-particles of the same type
        :rtype: :class:`Objects` class

        :Example:

        >>> print(events[0]["electron"].pick_charge(-1))
        +----------+--------+-------+--------+-------+------+------+-------+
        |  Object  |  eta   |  phi  |   PT   | jmass | ntrk | btag | hadem |
        +----------+--------+-------+--------+-------+------+------+-------+
        | electron | -0.745 | 4.253 | 286.72 |  0.0  | -1.0 | 0.0  |  0.0  |
        +----------+--------+-------+--------+-------+------+------+-------+
        >>> print(events[0]["electron"].pick_charge(+1))
        +----------+--------+-------+-------+-------+------+------+-------+
        |  Object  |  eta   |  phi  |   PT  | jmass | ntrk | btag | hadem |
        +----------+--------+-------+-------+-------+------+------+-------+
        | electron | -0.073 | 4.681 | 44.56 |  0.0  | 1.0  | 0.0  |  0.0  |
        +----------+--------+-------+-------+-------+------+------+-------+
        >>> print(events[0]["jet"].pick_charge(+1))
        +--------+-----+-----+----+-------+------+------+-------+
        | Object | eta | phi | PT | jmass | ntrk | btag | hadem |
        +--------+-----+-----+----+-------+------+------+-------+
        +--------+-----+-----+----+-------+------+------+-------+
        """

        objects = [object_ for object_ in self if object_.charge() == charge]

        return Objects(objects)

###############################################################################


class Event(dict):
    """
    A single LHCO event.

    Includes functions to parse a list of lines of a single LHCO event from an
    LHCO file into an :class:`Event` class.

    This class inherits the dictionary class - it is itself a dictionary with
    keys. Dictionary keys are objects that might be in an event, e.g.

    - :literal:`photon`
    - :literal:`electron`
    - :literal:`muon`
    - :literal:`tau`
    - :literal:`jet`
    - :literal:`MET`

    Each dictionary entry is itself an :class:`Objects` class - a class
    designed for a list of objects, e.g. all electrons in an event.

    :Example:

    >>> print(events[0])
    +----------+--------+-------+--------+-------+------+------+-------+
    |  Object  |  eta   |  phi  |   PT   | jmass | ntrk | btag | hadem |
    +----------+--------+-------+--------+-------+------+------+-------+
    | electron | -0.745 | 4.253 | 286.72 |  0.0  | -1.0 | 0.0  |  0.0  |
    | electron | -0.073 | 4.681 | 44.56  |  0.0  | 1.0  | 0.0  |  0.0  |
    |   jet    | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
    |   jet    | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
    |   jet    | 0.811  | 6.028 | 17.49  |  3.47 | 8.0  | 0.0  |  2.37 |
    |   jet    | 0.596  | 0.853 | 12.47  |  2.53 | 7.0  | 0.0  |  1.26 |
    |   jet    | -1.816 | 0.032 |  6.11  |  1.18 | 0.0  | 0.0  |  0.56 |
    |   jet    | 0.508  | 1.421 |  6.01  |  0.94 | 7.0  | 0.0  |  2.59 |
    |   MET    |  0.0   | 2.695 | 21.43  |  0.0  | 0.0  | 0.0  |  0.0  |
    +----------+--------+-------+--------+-------+------+------+-------+
    """

    def __init__(self, lines=None, dictionary=None, trigger_info=None):
        """
        :param lines: List of lines of LHCO event from an LHCO file
        :type lines: list
        :param dictionary: Dictionary or zipped lists for new dictionary
        :type dictionary: dict
        :param trigger_info: Tuple of event number and trigger word value
        :type trigger_info: tuple
        """

        self._lines = lines  # Save list of lines of whole event
        self.trigger_info = trigger_info

        if lines and dictionary:
            raise ValueError("Must specify lines or dictionary")

        # Ordinary initialisation
        if dictionary:
            super(self.__class__, self).__init__(dictionary)
            return
        else:
            super(self.__class__, self).__init__(EMPTY_DICT)

        # Build a dictionary of objects appearing in the event,
        # e.g. self["electron"] is initialised to be an empty Objects class
        for name in NAMES:
            self[name] = Objects()  # List of e.g. "electron"s in event

        # Check that it is a non-empty list
        if self._lines:
            self.__parse()  # Parse the event
            # Check whether agrees with LHCO file
            if self.__count_file() != self.count_parsed():
                warnings.warn("Inconsistent numbers of objects in event:\n %s" % str(self))
        else:
            warnings.warn("Adding empty event")

    def __str__(self):
        """
        Make a nice readable table format.

        :returns: Readable table format
        :rtype: :string
        """

        # Make table of event
        table = pt(HEADINGS)

        # Add rows to the table
        for name in NAMES:  # Iterate object types e.g electrons
            for object_ in self[name]:  # Iterate all objects of that type
                table.add_row(object_._row())

        return table.get_string()

    def number(self, anti_lepton=False):
        """
        Count objects of each type, e.g. :literal:`electron`.

        :param anti_lepton: Whether anti-leptons treated separately
        :type anti_lepton: bool

        :return: A dictionary of the numbers of objects of each type
        :rtype: dict

        :Example:

        >>> print(events[0])
        +----------+--------+-------+--------+-------+------+------+-------+
        |  Object  |  eta   |  phi  |   PT   | jmass | ntrk | btag | hadem |
        +----------+--------+-------+--------+-------+------+------+-------+
        | electron | -0.745 | 4.253 | 286.72 |  0.0  | -1.0 | 0.0  |  0.0  |
        | electron | -0.073 | 4.681 | 44.56  |  0.0  | 1.0  | 0.0  |  0.0  |
        |   jet    | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |   jet    | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        |   jet    | 0.811  | 6.028 | 17.49  |  3.47 | 8.0  | 0.0  |  2.37 |
        |   jet    | 0.596  | 0.853 | 12.47  |  2.53 | 7.0  | 0.0  |  1.26 |
        |   jet    | -1.816 | 0.032 |  6.11  |  1.18 | 0.0  | 0.0  |  0.56 |
        |   jet    | 0.508  | 1.421 |  6.01  |  0.94 | 7.0  | 0.0  |  2.59 |
        |   MET    |  0.0   | 2.695 | 21.43  |  0.0  | 0.0  | 0.0  |  0.0  |
        +----------+--------+-------+--------+-------+------+------+-------+
        >>> print(events[0].number())
        +--------+----------+------+-----+-----+-----+
        | photon | electron | muon | tau | jet | MET |
        +--------+----------+------+-----+-----+-----+
        |   0    |    2     |  0   |  0  |  6  |  1  |
        +--------+----------+------+-----+-----+-----+
        >>> print(events[0].number(anti_lepton=True))
        +--------+----------+------+-----+-----+-----+---------------+-----------+----------+
        | photon | electron | muon | tau | jet | MET | anti-electron | anti-muon | anti-tau |
        +--------+----------+------+-----+-----+-----+---------------+-----------+----------+
        |   0    |    1     |  0   |  0  |  6  |  1  |       1       |     0     |    0     |
        +--------+----------+------+-----+-----+-----+---------------+-----------+----------+
        """

        number = PrintDict()  # Dictionary class, with printing function

        number_names = list(NAMES)
        if anti_lepton:
            for name in LEPTON:
                anti_name = "anti-" + name
                number_names.append(anti_name)

        for name in number_names:
            number[name] = 0

        # Record number of objects in an event
        for name, objects in self.iteritems():
            if name in LEPTON and anti_lepton:
                number[name] = len(objects.pick_charge(-1))
                anti_name = "anti-" + name
                number[anti_name] = len(objects.pick_charge(1))
            else:
                number[name] = len(objects)

        return number

    def count_parsed(self):
        """
        Count total number of objects of all types, e.g. electrons.

        :returns: The total number of objects
        :rtype: int

        :Example:

        >>> print(events[0].count_parsed())
        9
        """
        return sum([len(objects) for objects in self.itervalues()])

    def __count_file(self):
        """
        Return number of objects in the event according to LHCO file.

        This attribute is intended to be private - i.e. only called from within
        this class itself.

        :returns: Number of events in objects in the event
        :rtype: int
        """
        return len(self._lines) - 1  # -1 for trigger information

    def add_object(self, name, dictionary=None):
        """
        Add an object to the event. Append a list element of :class:`Object`
        class.

        :param name: Name of object, e.g. :literal:`electron`
        :type name: string
        :param dictionary: Dictionary of object properties
        :type dictionary: dict
        """
        if name not in NAMES:
            warnings.warn("Name not recognised: %s" % name)

        self[name].append(Object(name, dictionary))

    def __parse(self):
        """
        Parse a list of lines of a single LHCO event into an :class:`Event`
        object.

        The LHCO format is described
        `here <http://madgraph.phys.ucl.ac.be/Manual/lhco.html>`_.

        This attribute is intended to be private - i.e. only called from within
        this class itself.
        """

        # Parse the event line by line as chunks of words
        for line in self._lines:

            words = line.split()
            number = words[0]  # Number of object in event

            if number is "0":  # "0" events are trigger information
                assert len(words) == 3, "Trigger should have 3 words"
                self.trigger_info = map(int, words[1:])
            else:
                assert len(words) == 11, "Event should have 11 words"

                # Map words to floats
                values = map(float, words)

                # The first two - # and typ - are integers
                values[0] = int(values[0])
                values[1] = int(values[1])

                index = values[1]  # Index of object in LHCO format
                name = NAMES_DICT[index]  # Name of object, e.g. "electron"

                # Append an Object with the LHCO properties
                self.add_object(name, zip(PROPERTIES, values))

    def __add__(self, other):
        """
        Add two events together, returning a new :class:`Event` class with all
        the e.g. electrons that were in original two events.

        :param other: An :class:`Event` class of objects in an event
        :type other: class:`Event`

        :returns: An :class:`Event` class of objects in a this event and \
        extra event
        :rtype: :class:`Event`

        :Example:

        >>> print(events[0] + events[1])
        +----------+--------+-------+--------+-------+------+------+-------+
        |  Object  |  eta   |  phi  |   PT   | jmass | ntrk | btag | hadem |
        +----------+--------+-------+--------+-------+------+------+-------+
        | electron | -0.745 | 4.253 | 286.72 |  0.0  | -1.0 | 0.0  |  0.0  |
        | electron | -0.073 | 4.681 | 44.56  |  0.0  | 1.0  | 0.0  |  0.0  |
        | electron | -2.153 | 4.152 | 36.66  |  0.0  | -1.0 | 0.0  |  0.0  |
        | electron | -0.95  | 0.521 |  41.2  |  0.0  | 1.0  | 0.0  |  0.0  |
        |   jet    | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |   jet    | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        |   jet    | 0.811  | 6.028 | 17.49  |  3.47 | 8.0  | 0.0  |  2.37 |
        |   jet    | 0.596  | 0.853 | 12.47  |  2.53 | 7.0  | 0.0  |  1.26 |
        |   jet    | -1.816 | 0.032 |  6.11  |  1.18 | 0.0  | 0.0  |  0.56 |
        |   jet    | 0.508  | 1.421 |  6.01  |  0.94 | 7.0  | 0.0  |  2.59 |
        |   jet    | -2.075 | 2.035 | 37.34  |  4.58 | 6.0  | 0.0  |  1.54 |
        |   jet    | 2.188  | 4.088 | 12.71  |  2.77 | 4.0  | 0.0  |  2.14 |
        |   MET    |  0.0   | 2.695 | 21.43  |  0.0  | 0.0  | 0.0  |  0.0  |
        |   MET    |  0.0   | 5.378 |  9.6   |  0.0  | 0.0  | 0.0  |  0.0  |
        +----------+--------+-------+--------+-------+------+------+-------+
        """
        combination = Event()
        for name in self.keys():  # e.g. "electron"
            combination[name] = self[name] + other[name]  # Add Objects() lists

        return combination

    def __mul__(self, other):
        """
        Multiplying returns an :class:`Event` class.

        :param other: Number to multiply by
        :type other: int

        :returns: Numerous copies of objects in :class:`Event` class
        :rtype: :class:`Event`

        :Example:

        >>> print(events[0] * 2)
        +----------+--------+-------+--------+-------+------+------+-------+
        |  Object  |  eta   |  phi  |   PT   | jmass | ntrk | btag | hadem |
        +----------+--------+-------+--------+-------+------+------+-------+
        | electron | -0.745 | 4.253 | 286.72 |  0.0  | -1.0 | 0.0  |  0.0  |
        | electron | -0.073 | 4.681 | 44.56  |  0.0  | 1.0  | 0.0  |  0.0  |
        | electron | -0.745 | 4.253 | 286.72 |  0.0  | -1.0 | 0.0  |  0.0  |
        | electron | -0.073 | 4.681 | 44.56  |  0.0  | 1.0  | 0.0  |  0.0  |
        |   jet    | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |   jet    | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        |   jet    | 0.811  | 6.028 | 17.49  |  3.47 | 8.0  | 0.0  |  2.37 |
        |   jet    | 0.596  | 0.853 | 12.47  |  2.53 | 7.0  | 0.0  |  1.26 |
        |   jet    | -1.816 | 0.032 |  6.11  |  1.18 | 0.0  | 0.0  |  0.56 |
        |   jet    | 0.508  | 1.421 |  6.01  |  0.94 | 7.0  | 0.0  |  2.59 |
        |   jet    | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |   jet    | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        |   jet    | 0.811  | 6.028 | 17.49  |  3.47 | 8.0  | 0.0  |  2.37 |
        |   jet    | 0.596  | 0.853 | 12.47  |  2.53 | 7.0  | 0.0  |  1.26 |
        |   jet    | -1.816 | 0.032 |  6.11  |  1.18 | 0.0  | 0.0  |  0.56 |
        |   jet    | 0.508  | 1.421 |  6.01  |  0.94 | 7.0  | 0.0  |  2.59 |
        |   MET    |  0.0   | 2.695 | 21.43  |  0.0  | 0.0  | 0.0  |  0.0  |
        |   MET    |  0.0   | 2.695 | 21.43  |  0.0  | 0.0  | 0.0  |  0.0  |
        +----------+--------+-------+--------+-------+------+------+-------+
        """
        prod = Event()
        for name in self.keys():  # e.g. "electron"
            prod[name] = self[name] * other  # Multiply object by object

        return prod

    def __rmul__(self, other):
        """ See :func:`__mul__`. """
        return self.__mul__(other)

    def LHCO(self, f_name):
        """
        Write event in LHCO format.

        :param f_name: Name of LHCO file to be written
        :type f_name: string
        """

        # Potentially useful in some circumstances to print
        # information in readable format. But generally slow and
        # makes big files.
        # print(comment(self), file=open(f_name, "a"))

        header = ["#", "number", "trig word"]
        header = [oo.ljust(10) for oo in header]
        print(*header, file=open(f_name, "a"))

        try:
            trigger = [0] + self.trigger_info
        except TypeError:
            warnings.warn("Missing trigger information")
            trigger = [0] * 3

        trigger = [repr(oo).ljust(10) for oo in trigger]
        print(*trigger, file=open(f_name, "a"))

        header = ["#", "typ", "eta", "phi", "pt",
                  "jmass", "ntrk", "btag", "had/em",
                  "dummy", "dummy"]
        header = [oo.ljust(10) for oo in header]
        print(*header, file=open(f_name, "a"))

        for name in NAMES:
            try:
                self[name].LHCO(f_name)
            except KeyError:
                warnings.warn("Missing key in events: %s" % name)

    def multiplicity(self):
        """
        Count total number of objects of all types, e.g. electrons,
        excluding MET.

        :returns: The total number of objects, excluding MET
        :rtype: int

        :Example:

        >>> print(events[0].number())
        +--------+----------+------+-----+-----+-----+
        | photon | electron | muon | tau | jet | MET |
        +--------+----------+------+-----+-----+-----+
        |   0    |    2     |  0   |  0  |  6  |  1  |
        +--------+----------+------+-----+-----+-----+
        >>> print(events[0])
        +----------+--------+-------+--------+-------+------+------+-------+
        |  Object  |  eta   |  phi  |   PT   | jmass | ntrk | btag | hadem |
        +----------+--------+-------+--------+-------+------+------+-------+
        | electron | -0.745 | 4.253 | 286.72 |  0.0  | -1.0 | 0.0  |  0.0  |
        | electron | -0.073 | 4.681 | 44.56  |  0.0  | 1.0  | 0.0  |  0.0  |
        |   jet    | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |   jet    | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        |   jet    | 0.811  | 6.028 | 17.49  |  3.47 | 8.0  | 0.0  |  2.37 |
        |   jet    | 0.596  | 0.853 | 12.47  |  2.53 | 7.0  | 0.0  |  1.26 |
        |   jet    | -1.816 | 0.032 |  6.11  |  1.18 | 0.0  | 0.0  |  0.56 |
        |   jet    | 0.508  | 1.421 |  6.01  |  0.94 | 7.0  | 0.0  |  2.59 |
        |   MET    |  0.0   | 2.695 | 21.43  |  0.0  | 0.0  | 0.0  |  0.0  |
        +----------+--------+-------+--------+-------+------+------+-------+
        """

        numbers = [self.number()[name] for name in NAMES if name is not "MET"]
        return sum(numbers)

    def delta_HT(self):
        r"""
        Minimize :math:`\Delta H_T` by dividing jets into partitions or
        pseudo-jets such that it is minimized.

        .. math::
            \Delta H_T = \sum_{j \in j_2} H_T - \sum_{j \in j_1} H_T

        This is a
        `partition problem <https://en.wikipedia.org/wiki/pp>`_.

        :returns: :math:`\Delta H_T`
        :rtype: float

        >>> events[0].delta_HT()
        3.3800000000000088
        """
        PT = [jet["PT"] for jet in self["jet"]]
        delta_HT = pp.solver(PT, algorithm=ALPHA_T_ALGORITHM)
        return delta_HT

    def alpha_T(self):
        r"""
        The :math:`\alpha_T` variable used in e.g. CMS searches for
        supersymmetry. See `arXiv:1303.2985 <http://arxiv.org/abs/1303.2985>`_.

        .. math::
            \alpha_T = \frac12 \frac{H_T - \Delta H_T}{\sqrt{H_T^2 - \Delta H_T^2}}

        There are various algorithms for calculating :math:`\Delta H_T`.
        See :func:`delta_HT`.

        :param algorithm: Choice of algorithm for :math:`\Delta H_T`
        :type algorithm: string

        :returns: :math:`\alpha_T` variable
        :rtype: float

        >>> events[0].alpha_T()
        1.4148163330213006
        """
        assert len(self["jet"]) > 1, "Calcuating alpha_T for event with one or no jets"

        HT = self.HT()
        MHT = self.MHT()
        delta_HT = self.delta_HT()
        return 0.5 * (HT - delta_HT) / (HT**2 - MHT**2)**0.5

    def mega_jets(self):
        r"""
        Divide :math:`n`-jets into two mega-jets as described in
        `arXiv:1502.00300 <http://arxiv.org/abs/1502.00300>`_.

        The choice of mega-jets minizes the sum of the invariant masses of
        the two mega-jets.

        This is a non-standard
        `partition problem <https://en.wikipedia.org/wiki/pp>`_.
        It differs from regular partition problems because we attempt to
        minimize a sum of invariant masses and invariant mass is non-linear.

        :returns: Fourmomentum of each mega-jet
        :rype: List of :class:`Fourmomentum`

        >>> for mega_jet in events[0].mega_jets():
        ...     print(mega_jet)
        ...     print(abs(mega_jet))
        +---------------+---------------+---------------+----------------+
        |       E       |      P_x      |      P_y      |      P_z       |
        +---------------+---------------+---------------+----------------+
        | 317.570707098 | 99.2274683859 | 269.239699011 | -118.794798267 |
        +---------------+---------------+---------------+----------------+
        66.3539290883
        +---------------+--------------+---------------+---------------+
        |       E       |     P_x      |      P_y      |      P_z      |
        +---------------+--------------+---------------+---------------+
        | 64.9644034641 | 32.129245428 | 11.1163896984 | 8.57454453438 |
        +---------------+--------------+---------------+---------------+
        54.6899293451
        """
        assert len(self["jet"]) > 1, "Attempting to combine one or no jets in mega-jets"

        all_jets = [jet.vector() for jet in self["jet"]]
        all_jets.sort(key=lambda jet: abs(jet), reverse=True)
        mass = lambda mega_jet_1, mega_jet_2: abs(mega_jet_1) + abs(mega_jet_2)
        mega_jets = pp.non_standard_solver(all_jets, mass, algorithm=RAZOR_ALGORITHM)
        return mega_jets

    def razor_MR(self):
        r"""
        The razor :math`M^R` variable, as described in
        `arXiv:1502.00300 <http://arxiv.org/abs/1502.00300>`_.

        .. math::
            M^R = sqrt{(|\vec j_1| + |\vec j_2|)^2 - (j_1^z j_2^z)^2}

        :returns: The razor :math`M_R` variable
        :rtype: float

        >>> events[0].razor_MR()
        327.57801830352366
        """
        jet_1, jet_2 = self.mega_jets()
        abs_vector = lambda jet: (jet[1]**2 + jet[2]**2 + jet[3]**2)**0.5
        MR = ((abs_vector(jet_1) + abs_vector(jet_2))**2 - (jet_1[3] + jet_2[3])**2)**0.5
        return MR

    def razor_MRT(self):
        r"""
        The razor :math`M^R_T` variable, as described in
        `arXiv:1502.00300 <http://arxiv.org/abs/1502.00300>`_.

        .. math::
            M^R_T = \sqrt{1/2} \sqrt{E_T j_T - E_x j_x - E_y j_y}

        :returns: The razor :math`M^R_T` variable
        :rtype: float

        >>> events[0].razor_MRT()
        57.353513060398633
        """
        jet_1, jet_2 = self.mega_jets()
        jet = jet_1 + jet_2
        MET = self["MET"][0].vector()
        MRT = (0.5 * (MET.PT() * jet.PT() - MET[1] * jet[1] - MET[2] * jet[2]))**0.5
        return MRT

    def razor_R(self):
        r"""
        The razor :math`R` variable, as described in
        `arXiv:1502.00300 <http://arxiv.org/abs/1502.00300>`_.

        .. math::
            R = M_T^R / M_R

        :returns: The razor :math`R` variable
        :rtype: float

        >>> events[0].razor_R()
        0.175083521652105
        """
        return self.razor_MRT() / self.razor_MR()

    def ET(self):
        r"""
        Calculate the scalar sum of transverse energy in an event:

        .. math::
            E_T = \sum p_T

        :returns: :math:`E_T`, scalar sum of transverse energy in an event
        :rtype: float

        :Example:

        >>> print(events[0].ET())
        661.76
        """

        ET = 0.
        no_MET_names = (name for name in NAMES if name is not "MET")
        for name in no_MET_names:
            ET += sum([object_["PT"] for object_ in self[name]])

        return ET

    def MET(self, LHCO=False):
        r"""
        Calculate the vector sum of transverse energy in an event:

        .. math::
            MET = |\sum \vec p_T|

        .. warning::
          Detector-simulators compute MET, which is stored in the \
          LHCO file. That might differ from :math:`|\sum \vec p_T|`.


        :param LHCO: Whether to read MET from the LHCO
        :type LHCO: bool

        :returns: MET, vector sum of transverse energy in an event
        :rtype: float

        :Example:

        >>> print(events[0].MET())
        21.363659506
        >>> print(events[0].MET(LHCO=True))
        21.43
        """

        if LHCO:
            MET = self["MET"][0]["PT"]
        else:
            MET_vector = Fourvector()
            no_MET_names = (name for name in NAMES if name is not "MET")
            for name in no_MET_names:
                MET_vector += sum([object_.vector() for object_ in self[name]])
            MET = MET_vector.PT()

        return MET

    def HT(self):
        r"""
        Calculate the scalar sum of transverse energy in jets in an event:

        .. math::
            H_T = \sum_j p_T

        :returns: :math:`H_T`, scalar sum of transverse energy in jets \
        in an event
        :rtype: float

        :Example:

        >>> print(events[0].HT())
        330.48
        """
        assert len(self["jet"]) > 0, "Calcuating HT for event with no jets"
        return sum([object_["PT"] for object_ in self["jet"]])

    def MHT(self):
        r"""
        Calculate the vector sum of transverse energy in jets in an event:

        .. math::
            MHT = |\sum_j \vec p_T|

        :returns: MHT, vector sum of transverse energy in jets in an event
        :rtype: float

        :Example:

        >>> print(events[0].MHT())
        309.603169784
        """
        assert len(self["jet"]) > 0, "Calcuating MHT for event with no jets"

        MHT_vector = sum([object_.vector() for object_ in self["jet"]])
        MHT = MHT_vector.PT()
        return MHT

    def number_b_jets(self):
        """
        Count the number of b-jets in an event.

        :returns: Number of b-jets
        :rtype: int

        :Example:

        >>> events[0].number_b_jets()
        0
        """

        return len(self.pick_b_jets())

    def pick_b_jets(self, tagged=True):
        """
        Find b-jets in an event.

        :param tagged: Pick b-jets that were tagged or untagged
        :type tagged: bool

        :returns: b-tagged jets (if tagged) or ordinary jets (if not tagged)
        :rtype: :class:`Objects` class

        :Example:

        >>> print(events[0].pick_b_jets())
        +--------+-----+-----+----+-------+------+------+-------+
        | Object | eta | phi | PT | jmass | ntrk | btag | hadem |
        +--------+-----+-----+----+-------+------+------+-------+
        +--------+-----+-----+----+-------+------+------+-------+
        >>> print(events[0].pick_b_jets(False))
        +--------+--------+-------+--------+-------+------+------+-------+
        | Object |  eta   |  phi  |   PT   | jmass | ntrk | btag | hadem |
        +--------+--------+-------+--------+-------+------+------+-------+
        |  jet   | -0.565 | 1.126 | 157.44 | 12.54 | 16.0 | 0.0  |  0.57 |
        |  jet   | -0.19  | 1.328 | 130.96 |  12.3 | 18.0 | 0.0  | 10.67 |
        |  jet   | 0.811  | 6.028 | 17.49  |  3.47 | 8.0  | 0.0  |  2.37 |
        |  jet   | 0.596  | 0.853 | 12.47  |  2.53 | 7.0  | 0.0  |  1.26 |
        |  jet   | -1.816 | 0.032 |  6.11  |  1.18 | 0.0  | 0.0  |  0.56 |
        |  jet   | 0.508  | 1.421 |  6.01  |  0.94 | 7.0  | 0.0  |  2.59 |
        +--------+--------+-------+--------+-------+------+------+-------+
        """
        objects = [jet for jet in self["jet"] if jet["btag"] == tagged]
        return Objects(objects)

###############################################################################


class Object(dict):
    """
    A single object in an LHCO event, e.g. a single electron.

    This object inherits the dictionary class - it is itself a dictionary. The
    keys correspond to an object's properties:

    - :literal:`event`
    - :literal:`type`
    - :literal:`eta`
    - :literal:`phi`
    - :literal:`PT`
    - :literal:`jmass`
    - :literal:`ntrk`
    - :literal:`btag`
    - :literal:`hadem`
    """

    def __init__(self, name=None, dictionary=None):
        """
        Initialise a single object, e.g. a single electron.

        :param name: Name of object, e.g. :literal:`electron`
        :type name: string
        :param dictionary: A dictionary, zipped lists etc for a new dictionary
        :type dictionary: dict
        """

        if name not in NAMES:
            warnings.warn("Name not recognised: %s" % name)

        # Ordinary initialisation
        if dictionary:
            super(self.__class__, self).__init__(dictionary)

        self.name = name  # Save name of object

    def __str__(self):
        """
        Make e.g. properties about an electron into an easy to read table
        format.

        :returns: String of :class:`Object` class
        :rtype: string
        """

        # Make table of event
        table = pt(HEADINGS)
        table.add_row(self._row())

        return table.get_string()

    def _row(self):
        """
        Make a row for object in an easy to read format.

        This attribute is intended to be semi-private - i.e. you are not
        encouraged to call this function directly.

        :returns: List of object's properties
        :rtype: list
        """

        row = [self[prop] for prop in PRINT_PROPERTIES]
        row.insert(0, self.name)

        return row

    def vector(self):
        r"""
        Make a four-momentum vector for this object.

        E.g., an electron's four-momentum from its
        :math:`P_T, \eta, \phi` parameters.
        :Example:

        :returns: Four-momentum vector for this object
        :rtype: :class:`Fourvector`

        >>> electron = Object()
        >>> electron["PT"] = 10
        >>> electron["eta"] = 1
        >>> electron["phi"] = 1
        >>> electron["jmass"] = 0.
        >>> print(electron.vector())
        +---------------+---------------+---------------+---------------+
        |       E       |      P_x      |      P_y      |      P_z      |
        +---------------+---------------+---------------+---------------+
        | 15.4308063482 | 5.40302305868 | 8.41470984808 | 11.7520119364 |
        +---------------+---------------+---------------+---------------+
        """
        return Fourvector_eta(self["PT"],
                              self["eta"],
                              self["phi"],
                              mass=self["jmass"])

    def LHCO(self, f_name):
        """
        Write object in LHCO format.

        :param f_name: Name of LHCO file to be written
        :type f_name: string
        """
        list_ = [repr(self[prop]).ljust(10) for prop in PROPERTIES]
        print(*list_, file=open(f_name, "a"))

    def charge(self):
        """
        Find charge of object.

        :returns: Charge of object. None if not electron, mu or tau
        :rtype: int

        :Example:

        >>> print(events[10]["electron"])
        +----------+--------+-------+-------+-------+------+------+-------+
        |  Object  |  eta   |  phi  |   PT  | jmass | ntrk | btag | hadem |
        +----------+--------+-------+-------+-------+------+------+-------+
        | electron | -1.359 | 1.056 | 79.51 |  0.0  | -1.0 | 0.0  |  0.02 |
        | electron | -0.327 |  2.84 | 26.27 |  0.0  | 1.0  | 0.0  |  0.01 |
        +----------+--------+-------+-------+-------+------+------+-------+
        >>> print(events[10]["electron"][0].charge())
        -1
        >>> print(events[10]["electron"][1].charge())
        1
        >>> print(events[10]["jet"][0].charge())
        None
        """
        if not self.name:
            warnings.warn("Cannot find charge of particle with no name")
            return None
        if self.name not in LEPTON:
            warnings.warn("Cannot find charge of particle that is not lepton")
            return None
        else:
            return int(np.sign(self["ntrk"]))

###############################################################################


class Fourvector(np.ndarray):
    r"""
    A four-vector, with relevant addition, multiplication etc. operations.

    Builds a four-vector from Cartesian co-ordinates. Defines Minkowski
    product, square, addition of four-vectors.

    Inherits a numpy array.

    Four-vector is *contravariant* i.e.

    .. math::
      p^\mu = (E, \vec p)

      g_{\mu\nu} = \textrm{diag}(1, -1, -1, -1)

    :Example:

    >>> x = [1,1,1,1]
    >>> p = Fourvector(x)
    >>> print(p)
    +---+-----+-----+-----+
    | E | P_x | P_y | P_z |
    +---+-----+-----+-----+
    | 1 |  1  |  1  |  1  |
    +---+-----+-----+-----+
    >>> print(2 * p - p)
    +---+-----+-----+-----+
    | E | P_x | P_y | P_z |
    +---+-----+-----+-----+
    | 1 |  1  |  1  |  1  |
    +---+-----+-----+-----+
    """

    metric = np.diag([1., -1., -1., -1.])  # Define metric

    def __new__(cls, v_list=None):
        """
        Make four-vector from Cartesian co-ordinates.

        :param v_list: Length 4 list of four-vector in Cartesian co-ordinates
        :type v_list: list

        :returns: Four-momentum array
        :rtype: numpy.array
        """
        if v_list is None:
            v_list = [0.] * 4  # Default is empty four-vector
        elif len(v_list) != 4:
            raise ValueError("Four-vector must be length 4!")

        return np.asarray(v_list).view(cls)

    def __mul__(self, other):
        r"""
        Multiply four-vectors with Minkowski product, returning a scalar.

        If one entry is in fact a float or an integer, regular multiplication,
        returning a new four-vector.

        .. math::
            x \cdot y = x^\mu y^\nu g_{\mu\nu}

            n \times x = n \times x^\mu

        :param other: A float of another four-vector
        :type other: float or :class:`Fourvector`

        :returns: This four-vector multiplied by argument
        :rtype: float or :class:`Fourvector`

        :Example:

        >>> x = [1,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p*p)
        -2.0
        """

        sign_g_00 = np.sign(self.metric[0, 0])

        # Four-vector multiplication i.e. Minkowski product
        if isinstance(other, Fourvector):
            product = self.dot(self.metric).dot(other)

            if not product:
                warnings.warn("Minkowski-product zero")
            elif sign_g_00 != np.sign(product):
                warnings.warn("Minkowski-product wrong sign")

            return product

        # Four-vector multiplied by a number
        elif isinstance(other, float) or isinstance(other, int):
            product = np.ndarray.__mul__(self, other)
            return Fourvector(product)

        else:
            raise ValueError("Unsupported multiplication: %s" % type(other))

    def __rmul__(self, other):
        """ Four-vector multiplication. See :func:`__mul__`. """
        return self.__mul__(other)

    def __pow__(self, power):
        r"""
        Raising a four-vector to a power.

        .. warning::
            Only even powers supported.

        .. math::
            x^2 = x^\mu x^\nu g_{\mu\nu}

        :param power: Power to raise (must be an even integer)
        :type power: int

        :returns: This four-vector raised to an even power
        :rtype: float

        :Example:

        >>> x = [1,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p**2)
        -2.0
        >>> print(p**8)
        16.0
        """
        if power % 2 or not isinstance(power, int):
            raise ValueError("Only even integer powers supported: %i" % power)

        return self.__mul__(self)**(power / 2)

    def __str__(self):
        """
        Make a table of the four-vector for nice printing.

        :returns: Table of the four-vector
        :rtype: string

        """

        headings = ["E", "P_x", "P_y", "P_z"]
        table = pt(headings)
        table.add_row(self.tolist())

        return str(table)

    def __abs__(self):
        r"""
        Absolute value of this four-vector:

        .. math::
            |x| = \sqrt{x_\mu x^\mu \times g_{00}}

        i.e. the square-root of the Minkowski square. The presence of
        :math:`g_{00}` insures that the argument is positive, regardless of
        the metric convention.

        :returns: Absolute value of this four-vector
        :rtype: float

        :Example:

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(abs(p))
        4.69041575982
        """

        sign_g_00 = np.sign(self.metric[0, 0])
        square = self * self
        if not square:
            warnings.warn("Minkowski-square zero")
            return 0.
        else:
            assert sign_g_00 == np.sign(square) and square, \
                "Minkowski-square wrong sign"

        return (sign_g_00 * square)**0.5

    def __getslice__(self, other1, other2):
        """
        If four-vector is sliced, return a regular numpy array.

        :returns: Sliced four-vector
        :rtype: numpy.array
        """
        return np.array(np.ndarray.__getslice__(self, other1, other2))

    def phi(self):
        r"""
        Find angle :math:`\phi` around beam line from [0., 2.*pi].

        .. math::
            \phi = \arctan(p_y, p_x)

        :returns: Angle around beam line, :math:`\phi` from [0., 2.*pi]
        :rtype: float

        :Example:

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p.phi())
        0.785398163397
        """

        phi = atan(self[2], self[1])
        assert 0. <= phi <= 2. * pi, r"Angle \phi not in [0., 2.*pi]"

        return phi

    def PT(self):
        r"""
        Find :math:`P_T` - transverse magnitude of vector.

        .. math:
            P_T = \sqrt{p_x^2 + p_y^2}

        :returns: Transverse magnitude of vector, math:`P_T`
        :rtype: float

        :Example:

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p.PT())
        1.41421356237
        """
        return (self[1]**2 + self[2]**2)**0.5

    def theta(self):
        r"""
        Find angle math: `\theta` between vector and beam line from [0., pi].

        .. math::
            \theta = \arctan(r/z)

        :returns: Angle math: `\theta` between vector and beam line \
        from [0., pi].
        :rtype: float

        :Example:

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p.theta())
        0.955316618125
        """
        transverse_length = self.PT()
        beam_length = self[3]
        theta = atan(transverse_length, beam_length)

        assert 0. <= theta <= pi, r"Angle \theta not in [0., \pi]"

        return theta

    def eta(self):
        r"""
        Find pseudo-rapidity.

        .. math::
            \eta = -\ln(\tan\theta/2)

        :returns: Pseudo-rapidity, :math:`\eta`
        :rtype: float

        :Example:

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p.eta())
        0.658478948462
        """
        theta = self.theta()
        eta = -np.log(np.tan(theta/2.))  # Definition of pseudo-rapidity
        return eta

    def boost(self, beta):
        r"""
        Boost four-vector into a new reference-frame.

        .. math::
            x^\prime = \Lambda(\beta) x

        :param beta: Numpy array of :math:`\vec\beta` for boost
        :type beta: np.array

        :returns: Fourvector class, self but boosted by beta
        :rtype: :class:`Fourvector`

        :Example:

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> beta = p.beta_rest()
        >>> print(p.boost(beta))
        +---------------+--------------------+--------------------+--------------------+
        |       E       |        P_x         |        P_y         |        P_z         |
        +---------------+--------------------+--------------------+--------------------+
        | 4.69041575982 | -6.73072708679e-16 | -6.69603261727e-16 | -6.66133814775e-16 |
        +---------------+--------------------+--------------------+--------------------+
        """

        # Boost parameters
        beta_norm = np.linalg.norm(beta)
        gamma = (1. - beta_norm**2)**-0.5

        # Three-by-three sub-block of matrix
        # . . . .
        # . * * *
        # . * * *
        # . * * *
        beta_matrix = np.identity(3)
        beta_matrix += np.outer(beta, beta) * (gamma - 1.) / beta_norm**2

        # Length-four top row of matrix
        # * * * *
        # . . . .
        # . . . .
        # . . . .
        row = np.array([gamma,
                        -gamma * beta[0],
                        -gamma * beta[1],
                        -gamma * beta[2]])

        # Length-three column of matrix
        # . . . .
        # * . . .
        # * . . .
        # * . . .
        col = np.array([-gamma * beta[0], -gamma * beta[1], -gamma * beta[2]])

        # Make \Lambda matrix by stacking sub-bloc, row and column
        lambda_ = np.column_stack([col, beta_matrix])
        lambda_ = np.row_stack([row, lambda_])

        # Apply boost to self by matrix multiplication
        primed = lambda_.dot(self)

        return Fourvector(primed)

    def gamma(self):
        r"""
        Find Lorentz factor :math:`\gamma` for this four-vector:

        .. math::
          \gamma = p_0 / |p|

        :returns: Lorentz factor for this four-vector
        :rtype: float

        :Example:

        >>> x = [5,1,0,0]
        >>> p = Fourvector(x)
        >>> print(p.gamma())
        1.02062072616

        """
        mass = abs(self)
        assert mass, "Mass must be > 0: %s" % mass
        gamma = self[0] / mass  # gamma = E / M

        return gamma

    def beta(self):
        r"""
        Find :math:`\beta = v / c` for this four-vector:

        :returns: :math:`\beta` for this four-vector.
        :rtype: float

        :Example:

        >>> x = [5,1,0,0]
        >>> p = Fourvector(x)
        >>> print(p.beta())
        0.2

        """

        beta = (1. - self.gamma()**-2)**0.5
        return beta

    def beta_rest(self):
        r"""
        Find :math:`\vec\beta` for Lorentz boost to a frame in which this
        four-vector is at rest.

        .. math::
            \vec\beta = \vec v / c

        :returns: :math:`\vec\beta` as numpy array, i.e. \
        :math:`(\beta_x, \beta_y, \beta_z)`
        :rtype: numpy.array

        :Example:

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p.beta_rest())
        [ 0.2  0.2  0.2]

        """
        beta_norm = self.beta()
        beta = self.unit_vector() * beta_norm

        return beta

    def unit_vector(self):
        r"""
        Find unit vector in 3-vector direction.

        .. math::
            \hat v = \vec v / |\vec v|

        :returns: Unit vector
        :rtype: numpy.array

        :Example:

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p.unit_vector())
        [ 0.57735027  0.57735027  0.57735027]
        """
        unit = self[1:] / (self[1]**2 + self[2]**2 + self[3]**2)**0.5
        return unit

###############################################################################


def Fourvector_eta(PT, eta, phi, mass=0.):
    r"""
    Builds a four-vector from :math:`p, \eta, \phi` co-ordinates.

    .. math::
          \theta = 2 \arctan\exp(-\eta))

          p = P_T / \sin\theta

          E = \sqrt{p^2 + m^2}

          v = (E, p\sin\theta\cos\phi, p\sin\theta\sin\phi, p\cos\theta)

    Convention is that z-direction is the beam line.

    :param PT: Transverse momentum, :math:`P_T`
    :type PT: float
    :param eta: Pseudo-rapidity, :math:`\eta = -\ln[\tan(\theta/2)]`, \
    with theta angle to beam axis
    :type eta: float
    :param phi: Azimuthal angle :math:`\phi`, angle around beam
    :type phi: float
    :param mass: Mass of particle
    :type mass: float

    :returns: Four-vector object
    :rtype: :class:`Fourvector`

    :Example:

    >>> eta = 1.
    >>> phi = 1.
    >>> PT = 10.
    >>> mass = 1.
    >>> p = Fourvector_eta(PT, eta, phi, mass=mass)
    >>> print(p)
    +---------------+---------------+---------------+---------------+
    |       E       |      P_x      |      P_y      |      P_z      |
    +---------------+---------------+---------------+---------------+
    | 15.4631751123 | 5.40302305868 | 8.41470984808 | 11.7520119364 |
    +---------------+---------------+---------------+---------------+
    >>> print(p.eta())
    1.0
    >>> print(p.phi())
    1.0
    >>> print(p.PT())
    10.0
    >>> print(abs(p))
    1.0
    """

    theta = 2. * np.arctan(np.exp(-eta))
    abs_momentum = PT / sin(theta)
    energy = (abs_momentum**2 + mass**2)**0.5
    four_momentum = [energy,
                     abs_momentum * sin(theta) * cos(phi),
                     abs_momentum * sin(theta) * sin(phi),
                     abs_momentum * cos(theta)]

    # Make new four-vector with calculated Cartesian co-ordinates
    return Fourvector(four_momentum)

###############################################################################


def delta_R(object_1, object_2):
    r"""
    Find the angular separation between two objects.

    .. math::
        \Delta R = \sqrt{\Delta \phi^2 + \Delta \eta^2}

    :param object_1: An Object, e.g. an electron
    :type object_1: :class:`Object`
    :param object_2: Second Object, e.g. jet
    :type object_2: :class:`Object`

    :returns: Angular separation between objects, :math:`\Delta R`
    :rtype: float

    :Example:

    >>> delta_R_12 = delta_R(events[0]["jet"][1], events[0]["jet"][2])
    >>> print(delta_R_12)
    1.87309282121
    """

    delta_phi = acute(object_1["phi"], object_2["phi"])
    delta_eta = object_1["eta"] - object_2["eta"]

    return (delta_eta**2 + delta_phi**2)**0.5

###############################################################################


def comment(object_):
    """
    Places :literal:`#` at the beginning of every line in a string or an
    object to be represented as a string.

    :param object_: String to be commented
    :type object_: Object with :func:`__str__`

    :returns: Commented string
    :rtype: string
    """
    return "# " + str(object_).replace("\n", "\n# ")

###############################################################################


def atan(y_length, x_length):
    r"""
    The :math:`\arctan` function

    .. math::
        \arctan(y / x)

    but with correct quadrant and from [0., 2.*pi].

    :param y_length: :math:`y`-length
    :type y_length: float
    :param x_length: :math:`x`-length
    :type x_length: float

    :returns: Angle from [0., 2.*pi]
    :rytpe: float

    >>> atan(1,1)
    0.78539816339744828
    >>> atan(-1,1)
    5.497787143782138
    >>> 2. * pi - atan(1,1)
    5.497787143782138
    >>> atan(1,-1)
    2.3561944901923448
    >>> 0.5 * pi + atan(1,1)
    2.3561944901923448
    >>> atan(-1,-1)
    3.9269908169872414
    >>> pi + atan(1,1)
    3.9269908169872414
    """

    return np.arctan2(y_length, x_length) % (2. * pi)

###############################################################################


def acute(phi_1, phi_2):
    r"""
    Find acute angle :math:`\Delta\phi` between two angles from [0., pi].

    :param phi_1: First angle, :math:`\phi_1`
    :type phi_1: float
    :param phi_2: Second angle, :math:`\phi_2`
    :type phi_2: float

    :returns: Acute angle from [0., pi]
    :rytpe: float

    >>> acute(2. * pi, 0.)
    0.0
    >>> acute(4. * pi - 0.1, 8. * pi + 0.1)
    0.20000000000000107
    """

    # Consider difference on [0., 2.*pi]
    delta_phi = (phi_1 - phi_2) % (2. * pi)

    # Consider acute angle
    delta_phi = min(delta_phi, 2. * pi - delta_phi)

    assert 0. <= delta_phi <= pi, r"Angle \Delta\phi not in [0., pi]"

    return delta_phi

###############################################################################

if __name__ == "__main__":
    import doctest
    doctest.testmod(extraglobs={'events': Events("example.lhco")})
