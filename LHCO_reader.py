#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

"""
============
Introduction
============

Read an `LHCO <http://madgraph.phys.ucl.ac.be/Manual/lhco.html>`_
(or ROOT) file, from e.g. `PGS <
http://www.physics.ucdavis.edu/~conway/research/software/pgs/pgs4-general.html
>`_ or `Delphes <https://cp3.irmp.ucl.ac.be/projects/delphes>`_ into
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
- :literal:`MET` (missing transerse energy)
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
"""

__author__ = "Andrew Fowlie"
__copyright__ = "Copyright 2015"
__credits__ = ["Luca Marzola"]
__license__ = "GPL"
__maintainer__ = "Andrew Fowlie"
__email__ = "Andrew.Fowlie@Monash.Edu.Au"
__status__ = "Production"

###############################################################################

import os
import sys
import warnings
import inspect

import matplotlib.pyplot as plt
import numpy as np

from math import pi
from numpy import cos, sin
from prettytable import PrettyTable as pt
from collections import OrderedDict

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

        It is possible to initialize an :class:`Events` class without a LHCO
        or a ROOT file, and later append events to the list.

        .. warning::
            Parsing a ROOT file requires
            (and attempts to import) :mod:`LHCO_converter`.

        :param f_name: Name of an LHCO or ROOT file, including path
        :type f_name: str
        :param list_: A list for initializing events
        :type list_: list
        :param cut_list: Cuts applied to events and their acceptance
        :type cut_list: list
        :param n_events: Number of events to read from LHCO file
        :type n_events: int
        :param description: Information about events
        :type description: string
        """

        # Ordinary initialization
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
            raise Exception("File does not exist: %s" % f_name)
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
                raise Exception("Unknown file extension: %s" % f_name)

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

        n_events = self.n_events
        with open(self.LHCO_name, 'r') as f:
            for line in f:

                line = line.lstrip()  # Remove leading/trailing spaces

                # Ignore empty lines
                if not line:
                    continue

                line_startswith = line[0]
                if line_startswith is "#":  # Ignore comments etc
                    continue
                elif line_startswith is "0":  # New event in file
                    # Parse previous event, if there is one
                    try:
                        self.add_event(event)  # Add Event class
                    except:
                        pass
                    event = [line]  # New event - reset event list
                else:
                    # If there is not a "0", line belongs to current event,
                    # not a new event - add it to event
                    try:
                        event.append(line)
                    except:
                        warnings.warn("Possibly an event did not start with 0")
                        event = [line]

                # Don't parse more than a particular number of events
                if n_events and len(self) == n_events:
                    warnings.warn("Didn't parse all LHCO events, by request")
                    return

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

        # Initalize dictionary for numbers of objects
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
            with open(self.LHCO_name, 'r') as f:
                for line in f:
                    if line.strip().startswith("##  Number of Event"):
                        exp_events = int(line.split(":")[1])
                        break
                    if line.strip().startswith("# | Number of events |"):
                        exp_events = int(line.split("|")[2])
                        break
        except:
            pass

        return exp_events

    def __add__(self, other):
        """
        Combine :class:`Event` classes.

        "Add" two :class:`Event` classes together, making a new :class:`Event`
        class with all the events from each :class:`Event` class.

        :Example:

        >>> events + events
        +------------------+-----------------------------+
        | Number of events | 20000                       |
        | Description      | example.lhco + example.lhco |
        +------------------+-----------------------------+
        """

        if not isinstance(other, Events):
            raise Exception("Can only add Events with other Events object")

        # Make a description
        try:
            d = self.description + " + " + other.description
        except:
            warnings.warngin("Couldn't find descriptions")
            d = None

        # New set of events in Events class
        combined = Events(list_=list.__add__(self, other), description=d)

        return combined

    def __str__(self):
        """
        Make a string of an :class:`Events` class.

        Rather than attempting to return thousands of events, return summary
        information about set of events.

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

        # Find combined acceptance of all cuts
        combined_acceptable = 1.

        for ii, (cut, acceptance) in enumerate(self.cut_list):

            # Inspect source code
            try:
                cut_string = inspect.getsource(cut).strip()
            except:
                warnings.warn("Did not inspect source. Probably running interactively")
                cut_string = "Cut %s" % ii

            table.add_row([cut_string, str(acceptance)])
            combined_acceptable *= acceptance

        # Add information about combined acceptance
        if self.cut_list:
            table.add_row(["Combined acceptance", str(combined_acceptable)])

        table.align = "l"
        return table.get_string()

    def __repr__(self):
        """
        Represent object as a brief description. Representation of such a
        big list is anyway unreadable. See :func:`__str__`.
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
        +------------------------------------------------+--------------+
        | Number of events                               | 8656         |
        | Description                                    | example.lhco |
        | tau = lambda event: event.number()["tau"] == 1 | 0.8656       |
        | Combined acceptance                            | 0.8656       |
        +------------------------------------------------+--------------+
        """

        len_org = len(self)  # Remember original length
        if not len_org:
            raise Exception("No events")

        # Loop list in reverse order, removing events if they fail the cut.
        # Must be reverse order, because otherwise indices are
        # shifted when events are removed from the list.
        iterator = reversed(list(enumerate(self)))

        for ii, event in iterator:
            if cut(event):
                del self[ii]  # NB del is much faster than remove

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
        +------------------------------------------+--------------+
        | Number of events                         | 10000        |
        | Description                              | example.lhco |
        | PT = lambda object_: object_["PT"] < 30. | 1.0          |
        | Combined acceptance                      | 1.0          |
        +------------------------------------------+--------------+
        """

        if name not in NAMES:
            warnings.warn("Name not recoginised: %s" % name)
            return

        for ii in range(len(self)):
            self[ii][name].cut_objects(cut)

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
            warnings.warn("Name not recoginised: %s" % name)
            return

        if prop not in PROPERTIES:
            warnings.warn("Property not recoginised: %s" % prop)
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
        123.39502178770948
        """

        if name not in NAMES:
            warnings.warn("Name not recoginised: %s" % name)
            return

        if prop not in PROPERTIES:
            warnings.warn("Property not recoginised: %s" % prop)
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

        if name not in NAMES:
            warnings.warn("Name not recoginised: %s" % name)
            return

        if prop not in PROPERTIES:
            warnings.warn("Property not recoginised: %s" % prop)
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
            raise Exception("Cannot overwrite %s" % LHCO_name)

        preamble = """LHCO file created with LHCO_reader (https://github.com/innisfree/LHCO_reader).
See http://madgraph.phys.ucl.ac.be/Manual/lhco.html for a description of the LHCO format."""
        print(comment(preamble), file=open(LHCO_name, "w"), end="\n\n")
        print(comment(self), file=open(LHCO_name, "a"), end="\n\n")

        for nn, event in enumerate(self):
            print("# Event number:", nn, file=open(LHCO_name, "a"))
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
            raise Exception("Could not convert to ROOT")

        return ROOT_name

###############################################################################


class PrintDict(OrderedDict):
    """
    An ordinary dictionary with a nice printing function.
    """
    def __str__(self):
        """
        Make an easy to read table of the dictionary contents.
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
            warnings.warn("Property not recoginised: %s" % prop)
            return

        # Simply sort the list by the required property
        self.sort(key=lambda obj: obj[prop], reverse=reverse)

        return self

    def __str__(self):
        """
        Return a nice readable table format.
        """

        # Make table of event
        table = pt(HEADINGS)

        # Add rows to the table
        for obj in self:
            table.add_row(obj._row())

        return table.get_string()

    def __add__(self, other):
        """
        Add :class:`Objects` together, returning a new :class:`Objects` class.

        E.g. you might wish to add :literal:`electron` with :literal:`muon` to
        make an :class:`Objects` class of all leptons.

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
            for oo in self:
                oo.LHCO(f_name)

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

        for ii, object_ in iterator:
            if cut(object_):
                del self[ii]  # NB del is much faster than remove

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

        objects = Objects()
        for object_ in self:
            if object_.charge() == charge:
                objects.append(object_)

        return objects

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
            raise("Must specify lines or dictionary")

        # Ordinary initialization
        if dictionary:
            super(self.__class__, self).__init__(dictionary)
            return
        else:
            super(self.__class__, self).__init__(EMPTY_DICT)

        # Build a dictionary of objects appearing in the event,
        # e.g. self["electron"] is initialized to be an empty Objects class
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
        """

        # Make table of event
        table = pt(HEADINGS)

        # Add rows to the table
        for name in NAMES:  # Iterate object types e.g electrons
            for obj in self[name]:  # Iterate all objects of that type
                table.add_row(obj._row())

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

        >>> print(Event().count_parsed())
        0
        """

        # Record number of objects in an event and keep total
        total_number = 0
        for objects in self.itervalues():
            total_number += len(objects)

        return total_number

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
            warnings.warn("Name not recoginised: %s" % name)

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
                self.trigger_info = map(int, words[1:])
                continue

            try:
                # Map words to floats
                values = map(float, words)

                # The first two - # and typ - are integers
                values[0] = int(values[0])
                values[1] = int(values[1])

                index = values[1]  # Index of object in LHCO format
                name = NAMES_DICT[index]  # Name of object, e.g. "electron"

                # Append an Object with the LHCO properties
                self.add_object(name, zip(PROPERTIES, values))
            except:
                warnings.warn("Couldn't parse line")

    def __add__(self, other):
        """
        Add two events together, returning a new :class:`Event` class with all
        the e.g. electrons that were in original two events.

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
        for key in self.keys():
            combination[key] = self[key] + other[key]  # Add Objects() lists

        return combination

    def __mul__(self, other):
        """
        Multiplying returns an :class:`Event` class.

        :param other: Number to multiply by
        :type other: int

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
        for key in self.keys():
            prod[key] = self[key] * other  # Multiply object by object

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
        except:
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

        # Record number of objects in an event and keep total
        multiplicity = 0
        for name in NAMES:
            if name is "MET":
                continue
            multiplicity += self.number()[name]

        return multiplicity

    def ET(self):
        """
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
        for name in NAMES:
            if name is "MET":
                continue
            for object_ in self[name]:
                ET += object_["PT"]

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
            for name in NAMES:
                if name is "MET":
                    continue
                for object_ in self[name]:
                    MET_vector += object_.vector()
            MET = MET_vector.PT()

        return MET

    def HT(self):
        """
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

        HT = 0.
        for object_ in self["jet"]:
            HT += object_["PT"]

        return HT

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

        MHT_vector = Fourvector()
        for object_ in self["jet"]:
            MHT_vector += object_.vector()

        MHT = MHT_vector.PT()
        return MHT

    def number_b_jets(self):
        """
        Count the number of b-jets in an event.

        :returns: Number of b-jets
        :rtype: integer

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

        objects = Objects()
        for jet in self["jet"]:
            if jet["btag"] == tagged:
                objects.append(jet)

        return objects

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
        Initialize a single object, e.g. a single electron.

        :param name: Name of object, e.g. :literal:`electron`
        :type name: string
        :param dictionary: A dictionary, zipped lists etc for a new dictionary
        :type dictionary: dict
        """

        if name not in NAMES:
            warnings.warn("Name not recoginised: %s" % name)

        # Ordinary initialization
        if dictionary:
            super(self.__class__, self).__init__(dictionary)

        self.name = name  # Save name of object

    def __str__(self):
        """
        Make e.g. properties about an electron into an easy to read table
        format.
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
        """

        row = [self.name]
        for prop in PRINT_PROPERTIES:
            row.append(self[prop])

        return row

    def vector(self):
        """
        Make a four-momentum vector for this object.

        E.g., an electron's four-momentum from its
        :math:`P_T, \eta, \phi` parameters.
        :Example:

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

        :returns: Charge of object. None if not eletron, mu or tau
        :rtype: integer

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
    product, square, additon of four-vectors.

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

    def __new__(self, v=None):
        """
        Make four-vector from Cartesian co-ordinates.

        :param v: Length 4 list of four-vector in Cartesian co-ordinates
        :type v: list
        """
        if v is None:
            v = [0.] * 4  # Default is empty four-vector
        elif len(v) != 4:
            raise("Four-vector must be length 4!")

        return np.asarray(v).view(self)

    def __mul__(self, other):
        r"""
        Multiply four-vectors with Minkowski product, returning a scalar.

        If one entry is in fact a float or an integer, regular multiplication,
        returning a new four-vector.

        .. math::
            x \cdot y = x^\mu y^\nu g_{\mu\nu}

            n \times x = n \times x^\mu

        :Example:

        >>> x = [1,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p*p)
        -2.0
        """

        # Four-vector multiplication i.e. Minkowksi product
        if isinstance(other, Fourvector):
            prod = 0.
            for ii in range(4):
                for jj in range(4):
                    prod += self[ii] * other[jj] * self.metric[ii, jj]
            return prod
        # Four-vector multiplied by a number
        elif isinstance(other, float) or isinstance(other, int):
            prod = np.ndarray.__mul__(self, other)
            return Fourvector(prod)

        else:
            raise Exception("Unsupported multiplication: %s" % type(other))

    def __rmul__(self, other):
        """ Four-vector multiplication. See :func:`__mul__`. """
        return self.__mul__(other)

    def __pow__(self, power):
        r"""
        Raising a four-vector to a power.

        .. warning::
            Only power 2 (Minkowski square) supported.

        .. math::
            x^2 = x^\mu x^\nu g_{\mu\nu}

        :param power: Power to raise
        :type power: int

        :Example:

        >>> x = [1,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p**2)
        -2.0
        """
        if not power == 2:
            raise Exception("Only power 2 is supported: %i" % power)

        return self.__mul__(self)

    def __str__(self):
        """ Make a table of the four-vector for nice printing. """

        headings = ["E", "P_x", "P_y", "P_z"]
        table = pt(headings)
        table.add_row(self.tolist())

        return str(table)

    def __abs__(self):
        r"""
        Absolute value of a four-vector:

        .. math::
            |x| = \sqrt{x_\mu x^\mu}

        i.e. the square-root of the Minkowski square.

        :Example:

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(abs(p))
        4.69041575982
        """
        return self.__mul__(self)**0.5

    def __getslice__(self, other1, other2):
        """
        If four-vector is sliced, return a regular numpy array.
        """
        return np.array(np.ndarray.__getslice__(self, other1, other2))

    def phi(self):
        r"""
        Find angle :math:`\phi` around beam line.

        .. math::
            \phi = \arctan(p_y/p_x)

        :Example:

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p.phi())
        0.785398163397
        """
        tan_phi = self[2] / self[1]
        phi = np.arctan(tan_phi)
        return phi

    def PT(self):
        r"""
        Return :math:`P_T` - transverse magnitude of vector.

        .. math:
            P_T = \sqrt{p_x^2 + p_y^2}

        :Example:

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p.PT())
        1.41421356237
        """
        return (self[1]**2 + self[2]**2)**0.5

    def theta(self):
        r"""
        Find angle math: `\theta` between vector and beam line.

        .. math::
            \theta = \arctan(r/z)

        :Example:

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p.theta())
        0.955316618125
        """
        r = self.PT()
        z = self[3]
        tan_theta = r / z
        theta = np.arctan(tan_theta)
        if z < 0:
            theta += pi
        return theta

    def eta(self):
        r"""
        Find pseudo-rapidity.

        .. math::
            \eta = -\ln(\tan\theta/2)

        :Example:

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p.eta())
        0.658478948462
        """
        theta = self.theta()
        eta = -np.log(np.tan(theta/2.))  # Definiton of pseudo-rapidity
        return eta

    def boost(self, beta):
        r"""
        Boost four-vector into a new refrence-frame.

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

        gamma = self[0] / abs(self)  # gamma = E / M
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
    p = PT / sin(theta)
    E = (p**2 + mass**2)**0.5
    v = [E,
         p * sin(theta) * cos(phi),
         p * sin(theta) * sin(phi),
         p * cos(theta)]

    # Make new four-vector with calculated Cartesian co-ordinates
    return Fourvector(v)

###############################################################################


def delta_R(o1, o2):
    r"""
    Find the angular separation between two objects.

    .. math::
        \Delta R = \sqrt{\Delta \phi^2 + \Delta \eta^2}

    :param o1: An Object, e.g. an electron
    :type o1: :class:`Object`
    :param o2: Second Object, e.g. jet
    :type o2: :class:`Object`

    :returns: Angular separation between objects, :math:`\Delta R`
    :rtype: float

    :Example:

    >>> delta_R_12 = delta_R(events[0]["jet"][1], events[0]["jet"][2])
    >>> print(delta_R_12)
    4.80541371788
    """
    delta_R = ((o1["eta"] - o2["eta"])**2 + (o1["phi"] - o2["phi"])**2)**0.5

    return delta_R

###############################################################################


def comment(x):
    """
    Places :literal:`#` at the beginning of every line in a string or an
    object to be represented as a string.

    :param x: String to be commented
    :type x: string

    :returns: Commented string
    :rtype: string
    """
    return "# " + str(x).replace("\n", "\n# ")


###############################################################################


if __name__ == "__main__":
    import doctest
    doctest.testmod(extraglobs={'events': Events("example.lhco")})
