#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

"""
Read an LHCO file, from e.g. PGS, into a convenient format of classes.

The code is object-oriented. A LHCO file is parsed into several layers:
- Events (inherits list): List of events
- Event (inherits dictionary): Dictionary of objects in an event,
e.g. "electron"
- Objects (inherits list): List of objects of a particular type in an event,
e.g. list of electrons
- Object (inherits dictionary): Dictionary of properties of an object,
e.g. an electron's transverse momentum

Simple usage is e.g.

>>> events = Events("example.lhco")
>>> print(events[11]["electron"][0]["PT"])
49.07

that code loads an LHCO file, and prints the transverse momentum of the first
electron in the eleventh event. The object keys are
- electron
- muon
- tau
- jet
- MET
- photon

The property keys from the LHCO file are
- event
- type
- eta
- phi
- PT
- jmass
- ntrk
- btag
- hadem
These properties are floats.

We add an additional property, a function, vector(), which returns a
four-momentum object.
"""

__author__ = "Andrew Fowlie"
__copyright__ = "Copyright 2014"
__credits__ = ["Luca Marzola"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Andrew Fowlie"
__email__ = "andrew.fowlie@kbfi.ee"
__status__ = "Production"

###############################################################################

import os
import sys
from prettytable import PrettyTable as pt
import numpy as np
from numpy import cos, sin
import warnings
import matplotlib.pyplot as plt
from collections import OrderedDict
from math import pi
import inspect
import copy

###############################################################################

# Classes for storing LHCO file data

###############################################################################


class Events(list):
    """
    All events in LHCO file as a list of "Event" objects.

    Functions to parse an LHCO file into a list of "Event" objects. This class
    inherits the list class; it is itself a list with an integer index. Each
    entry in the list is an event. Simple usuage e.g.

    >>> events=Events("example.lhco")
    >>> print(events)
    +------------------+--------------+
    | Number of events |    10000     |
    |       File       | example.lhco |
    +------------------+--------------+
    >>> print(events[100])
    +----------+-------+-------+-------+-------+------+------+-------+
    |  Object  |  eta  |  phi  |   PT  | jmass | ntrk | btag | hadem |
    +----------+-------+-------+-------+-------+------+------+-------+
    | electron | -2.14 | 1.816 | 16.76 |  0.0  | -1.0 | 0.0  |  0.01 |
    | electron | 1.183 | 4.001 | 15.84 |  0.0  | 1.0  | 0.0  |  0.0  |
    |  photon  |  0.13 | 2.109 | 16.68 |  0.0  | 0.0  | 0.0  |  0.05 |
    |  photon  | 0.217 | 6.149 |  5.7  |  0.0  | 0.0  | 0.0  |  0.08 |
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

    Each list entry is itself an Event class - a class designed to store
    a single event.
    """

    def __init__(self, f_name=None, list_=None, cut_list=None):
        """
        Parse an LHCO file into a list of Event objects.

        It is possible to initialize an Events class without a LHCO file,
        and later append events to the list.

        Arguments:
        f_name -- Name of an LHCO file, including path
        list_ -- A list for initalizing Events
        cut_list -- Cuts applied to events and their acceptance
        """
        
        # Ordinary initialization
        if list_:
            super(self.__class__, self).__init__(list_) 
             
        self.f_name = f_name  # Save file name

        if cut_list:
            self.cut_list = cut_list  # Save cuts
        else:
            self.cut_list = []

        if not self.f_name:
            warnings.warn("Events class without LHCO file")
        elif not os.path.isfile(self.f_name):  # Check that file exists
            raise Exception("File does not exist")
        else:
            self.__parse()  # Parse file
            if not len(self):  # Check whether any events were parsed
                warnings.warn("No events were parsed")
            elif not self.__exp_number():  # Check number of parsed events
                warnings.warn("Couldn't read total number of events from LHCO file")
            elif not self.__exp_number() == len(self):
                warnings.warn("Events were not parsed")

    def add_event(self, event):
        """
        Add an Event object to the Events class.

        Append the Events list with a new Event object.

        Arguments:
        event -- List of lines in LHCO file for event
        """
        self.append(Event(event))

    def __parse(self):
        """
        Parse an LHCO file into individual Event objects.

        Our strategy is to loop over file, line by line, and split the
        file into indiviual events. A new event begins with a "0" and ends
        an exising event.

        LHCO format - http://madgraph.phys.ucl.ac.be/Manual/lhco.html

        This attribute is intended to be private - i.e. only called from within
        this class itself.
        """

        event = []  # List of all lines of a single event from LHCO file

        with open(self.f_name, 'r') as f:
            for line in f:

                line = line.strip()  # Remove leading/trailing spaces

                if line.startswith("#") or not line:  # Ignore comments etc
                    continue
                elif line.startswith("0"):  # New event in file
                    # Parse previous event, if there is one
                    if event:
                        self.add_event(event)  # Add Event class
                    event = [line]  # New event - reset event list
                else:
                    # If there is not a "0", line belongs to current event,
                    # not a new event - add it to event
                    event.append(line)

            # Parse final event in file - because there isn't a following event
            # the final event won't be parsed as above
            self.add_event(event)

    def number(self):
        """
        Count the total numbers of objects in event, e.g. total number
        of electrons, jets, muons etc.

        Store the information in a dictionary.

        >>> print(events.number())
        +------+-------+------+--------+-------+----------+---------------+--------------+
        | tau  |  jet  | muon | photon |  MET  | electron | Total objects | Total events |
        +------+-------+------+--------+-------+----------+---------------+--------------+
        | 1512 | 33473 |  3   |  1350  | 10000 |  17900   |     64238     |    10000     |
        +------+-------+------+--------+-------+----------+---------------+--------------+

        Returns:
        number -- Dictionary of numbers of each object, indexed by e.g.
        "electron"
        """

        # Increment numbers of objects by looping through events
        number = PrintDict()
        for event in self:
            for key, objects in event.iteritems():
                if number.get(key):
                    number[key] += len(objects)
                else:
                    number[key] = len(objects)

        number["Total objects"] = sum(number.values())
        number["Total events"] = len(self)

        return number

    def __exp_number(self):
        """
        Returns expected number of events in LHCO file.

        Number of events written in the LHCO file, as "##  Number of Event : ".

        This attribute is intended to be private - i.e. only called from within
        this class itself.

        Returns:
        exp_events -- Number of events expected in LHCO file
        """

        # Read expected number of events from LHCO file
        exp_events = None

        try:
            with open(self.f_name, 'r') as f:
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
        Combine Event classes.

        "Add" two Event classes together, making a new Event class with all the
        events from each Event class.

        >>> events + events
        +------------------+-------+
        | Number of events | 20000 |
        |       File       |  None |
        +------------------+-------+
        """

        if not isinstance(other, Events):
            raise Exception("Can only add Events with other Events object")

        # New set of events in Events class
        combined = Events(list_=list.__add__(self, other))

        return combined

    def __str__(self):
        """
        String an Events class.

        Rather than attempting to print thousands of events, print summary
        information about set of events.

        >>> print(events)
        +------------------+--------------+
        | Number of events |    10000     |
        |       File       | example.lhco |
        +------------------+--------------+
        """

        table = pt(header=False)
        table.add_row(["Number of events", len(self)])
        table.add_row(["File", self.f_name])
        for cut, acceptance in self.cut_list:
            cut_string = inspect.getsource(cut).strip()
            table.add_row([cut_string, str(acceptance)])

        return table.get_string()

    def __repr__(self):
        """
        Represent object as a brief description.

        Representation of such a big list is anyway unreadable.
        """
        return self.__str__()

    def cut(self, cut):
        """
        Apply a cut.

        Arguments:
        cut -- Cut to apply to events

        >>> events = Events("example.lhco")
        >>> tau = lambda event: event.number()["tau"] == 1
        >>> events.cut(tau)
        >>> print(events)
        +------------------------------------------------+--------------+
        |                Number of events                |     8656     |
        |                      File                      | example.lhco |
        | tau = lambda event: event.number()["tau"] == 1 |    0.8656    |
        +------------------------------------------------+--------------+
        """

        self_org = copy.deepcopy(self)
        for event in self_org:
            if cut(event):
                self.remove(event)

        # Append information about the cut to events
        acceptance = len(self) / len(self_org)
        self.cut_list.append([cut, acceptance])

    def column(self, key, prop):
        """
        Make a list of all e.g. electron's transverse momentum.

        Arguments:
        key -- Type of object, e.g. electron
        prop -- Property of object, e.g. PT, transverse
        momentum

        Returns:
        column -- List of all e.g. electron's PT
        """

        # Make a list of the desired property
        column = []
        for event in self:
            for objects in event[key]:
                column.append(objects[prop])

        return column

    def mean(self, key, prop):
        """
        Find mean of e.g. electron's transverse momentum.

        Arguments:
        key -- Type of object, e.g. electron
        prop -- Property of object, e.g. PT, transverse
        momentum

        Returns:
        mean -- Mean of e.g. electron's PT

        >>> events.mean("electron", "PT")
        123.39502178770948
        """
        mean = np.mean(self.column(key, prop))

        return mean

    def plot(self, key, prop):
        """
        Show a 1-dimensional histogram of an object's property.

        Plot basic histogram with matplotlib. with crude titles and axis
        labels. The histogram is normalised to one.

        Arguments:
        key -- Type of object, e.g. electron
        prop -- Property of object for histogram, e.g. PT, transverse
        momentum

        >>> events.plot("electron", "PT")
        """
        data = self.column(key, prop)  # Make data into a column

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(data, 50, normed=1, facecolor='Crimson', alpha=0.9)
        ax.grid()  # Add grid lines

        ax.set_title(key)  # Titles etc
        ax.set_xlabel(prop)
        ax.set_ylabel("Frequency")
        plt.rc('font', size=20)  # Fonts - LaTeX possible, but slow

        plt.show()  # Show the plot on screen

    def __getslice__(self, i, j):
        """
        Slicing returns an Events class rather than a list.

        >>> print(events[:100])
        +------------------+--------------------------------+
        | Number of events |              100               |
        |       File       | Events 0 to 99 in example.lhco |
        +------------------+--------------------------------+
        """
        events = Events(list_=list.__getslice__(self, i, j))
        events.f_name = "Events {} to {} in {}".format(i, j-1, self.f_name)
        return events

    def __mul__(self, other):
        """
        Multiplying returns an Events class rather than a list.
        >>> print(events * 5)
        +------------------+--------------------------+
        | Number of events |          50000           |
        |       File       | 5 copies of example.lhco |
        +------------------+--------------------------+
        """
        events = Events(list_=list.__mul__(self, other))
        events.f_name = "{} copies of {}".format(other, self.f_name)
        return events

    def __rmul__(self, other):
        """ See __mul__. """
        return self.__mul__(other)

    def LHCO(self, file_name):
        """
        Write events in LHCO format.

        Arguments:
        file_name -- Name of LHCO file to be written

        >>> events[:10].LHCO("test.lhco")
        >>> same_events = Events("test.lhco")
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

        preamble = """LHCO file created with LHCO_reader (https://github.com/innisfree/LHCO_reader).
See http://madgraph.phys.ucl.ac.be/Manual/lhco.html for a description of the LHCO format."""
        print(comment(preamble), file=open(file_name, "w"), end="\n\n")
        print(comment(self), file=open(file_name, "a"), end="\n\n")

        for nn, event in enumerate(self):
            print("# Event number:", nn, file=open(file_name, "a"))
            event.LHCO(file_name)

###############################################################################


class Event(dict):
    """
    A single LHCO event.

    Parse a list of lines of a single LHCO event from an LHCO file into a
    single Event class. This class inherits the dictionary class - it is itself
    a dictionary with keys.

    Dictionary keys are objects that might be in an event, e.g.
    - photon
    - electron
    - muon
    - tau
    - jet
    - MET

    Each dicionary entry is iteself an Objects class - a class designed for a
    list of objects, e.g. all electrons in an event.

    >>> print(Event()["electron"])
    +--------+-----+-----+----+-------+------+------+-------+
    | Object | eta | phi | PT | jmass | ntrk | btag | hadem |
    +--------+-----+-----+----+-------+------+------+-------+
    +--------+-----+-----+----+-------+------+------+-------+
    """

    def __init__(self, lines=None, dictionary=None, trigger_info=None):
        """
        Parse a string from an LHCO file into a single event.

        Arguments:
        string -- String of single LHCO event from an LHCO file
        dictionary -- Dictionary or zipped lists for new dictionary
        trigger_info -- Tuple of event number and trigger word value
        """
        
        if lines and dictionary:
            raise("Must specify lines OR dictionary.")
        
        # Ordinary initialization
        if dictionary: 
            super(self.__class__, self).__init__(dictionary) 
            return
        
        self._lines = lines  # Save list of lines of whole event
        self.trigger_info = trigger_info

        # Dictionary of object names that appear in event, index corresponds
        # to the number in the LHCO convention. Dictionary rather than list,
        # because number 5 is missing
        self._names = {"0": "photon",
                       "1": "electron",
                       "2": "muon",
                       "3": "tau",
                       "4": "jet",
                       "6": "MET"
                       }

        # Build a dictionary of objects appearing in the event,
        # e.g. self["electron"] is initalized to be an empty Objects class
        for name in self._names.itervalues():
            self[name] = Objects()  # List of e.g. "electron"s in event

        # Check that the string is indeed a non-empty list
        if isinstance(self._lines, list) and self._lines:
            self.__parse()  # Parse the event
            # Check whether agrees with LHCO file
            if self.__count_number() != sum(self.number().values()):
                warnings.warn("Inconsistent numbers of objects in event:\n" + str(self))
        else:
            warnings.warn("Adding empty event")

    def __str__(self):
        """
        String an event into a nice readable format.
        """

        # Make table of event
        headings = ["Object"] + Object()._print_properties
        table = pt(headings)

        # Add rows to the table
        for name in self._names.itervalues():  # Iterate object types e.g electrons
            for obj in self[name]:  # Iterate all objects of that type
                table.add_row(obj._row())

        return table.get_string()

    def number(self):
        """
        Count objects of each type, e.g. electron.

        Returns:
        number -- A dictionary of the numbers of objects of each type

        >>> print(Event().number())
        +-----+-----+------+--------+-----+----------+
        | tau | jet | muon | photon | MET | electron |
        +-----+-----+------+--------+-----+----------+
        |  0  |  0  |  0   |   0    |  0  |    0     |
        +-----+-----+------+--------+-----+----------+
        """

        # Record number of objects in an event and keep total
        number = PrintDict()  # Dictionary class, with printing function
        for name, objects in self.iteritems():
            number[name] = len(objects)

        return number

    def __count_number(self):
        """
        Return number of objects in the event - according to LHCO file, first
        word of the last line.

        Returns:
        number (int) -- Number of events in objects in the event.
        """
        return int(self._lines[-1].split()[0])

    def add_object(self, name, dictionary=None):
        """
        Add an object to the event. Append a list element of Object class.

        Arguments:
        name -- Name of object, e.g. "electron"
        dictionary --- Dictionary of object properties
        """
        self[name].append(Object(name, dictionary))

    def __parse(self):
        """
        Parse a list of lines of a single LHCO event into an Event object.

        The LHCO format is http://madgraph.phys.ucl.ac.be/Manual/lhco.html

        We ignore the "0" line, which contains trigger information. We save
        all other information.

        This attribute is intended to be private - i.e. only called from within
        this class itself.
        """

        properties = Object()._properties  # Expected properties of object
        names = self._names

        for line in self._lines:  # Parse the event line by line

            words = line.split()
            number = words[0].strip()  # Number of object in event

            if number is "0":  # "0" events are trigger information
                self.trigger_info = map(int, words[1:])
                continue

            try:
                # Split line into individual properties
                values = map(float, words)
                # The first two - # and typ - are integers
                values[0] = int(values[0])
                values[1] = int(values[1])
                index = words[1].strip()  # Index of object in LHCO format
                name = names[index]  # Name of object, e.g. "electron"
                # Append an Object with the LHCO properties
                self.add_object(name, zip(properties, values))
            except:
                warnings.warn("Couldn't parse line:\n" + line)

    def __add__(self, other):
        """
        Add two events together.

        Adds two events, returning a new Event class with all the e.g.
        electrons that were in original two events.

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
        Multiplying returns an Event class.
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
        """ See __mul__. """
        return self.__mul__(other)

    def LHCO(self, file_name):
        """
        Write event in LHCO format.

        Arguments:
        file_name -- Name of LHCO file to be written
        """

        print(comment(self), file=open(file_name, "a"))

        header = ["#", "number", "trig word"]
        header = [oo.ljust(10) for oo in header]
        print(*header, file=open(file_name, "a"))

        trigger = [0] + self.trigger_info
        trigger = [repr(oo).ljust(10) for oo in trigger]
        print(*trigger, file=open(file_name, "a"))

        header = ["#", "typ", "eta", "phi", "pt", "jmass", "ntrk", "btag", "had/em", "dummy", "dummy"]
        header = [oo.ljust(10) for oo in header]
        print(*header, file=open(file_name, "a"))

        for name in self._names.itervalues():
            self[name].LHCO(file_name)

###############################################################################


class PrintDict(OrderedDict):
    """
    An ordinary dictionary with a nice printing function.
    """
    def __str__(self):
        """
        String the dictionary to an easy to read table.
        """

        # Make table of dictionary keys and entries
        table = pt(self.keys())
        table.add_row(self.values())

        return table.get_string()

###############################################################################


class Objects(list):
    """
    Objects in an LHCO event of a particular type.

    E.g., a list of all electron objects in an LHCO event. Each indiviudal
    electron object is an Object class.
    """

    def order(self, prop):
        """
        Order objects by a particular property, e.g. order all jets by
        transverse momentum, PT.

        The objects are listed in reverse order, biggest to smallest. E.g., if
        sorted by PT, the hardest jet appears first.

        Arguments:
        prop -- Property by which to sort objects, e.g. sort by "PT"

        Returns:
        self - List of objects, now ordered
        """

        if not self[0].get(prop):
            warnings.warn("Property not recoginised: " + prop)
            return

        # Simply sort the list in reverse order, e.g. if sorted by PT,
        # hardest jet is first
        self.sort(key=lambda obj: obj[prop], reverse=True)

        return self

    def __str__(self):
        """
        String Objects into a nice readable format.
        """

        # Make table of event
        headings = ["Object"] + Object()._print_properties
        table = pt(headings)

        # Add rows to the table
        for obj in self:
            table.add_row(obj._row())

        return table.get_string()

    def __add__(self, other):
        """
        Add Objects together, returning a new Objects class.

        E.g. you might wish to add "electron" with "muon" to make an Objects
        class of all leptons.

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
        +------+--------------+-------+------+--------+-------+----------+---------------+--------------+
        | tau  | jet+electron |  jet  | muon | photon |  MET  | electron | Total objects | Total events |
        +------+--------------+-------+------+--------+-------+----------+---------------+--------------+
        | 1512 |    51373     | 33473 |  3   |  1350  | 10000 |  17900   |     115611    |    10000     |
        +------+--------------+-------+------+--------+-------+----------+---------------+--------------+
        """
        combination = Objects(list.__add__(self, other))

        return combination

    def __getslice__(self, i, j):
        """
        Slicing returns an Objects class rather than a list.

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
        Multiplying returns an Objects class rather than a list.

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
        """ See __mul__. """
        return self.__mul__(other)

    def LHCO(self, file_name):
        """
        Write objects in LHCO format.

        Arguments:
        file_name -- Name of LHCO file to be written
        """
        if self:
            self.order("PT")
            for oo in self:
                oo.LHCO(file_name)

###############################################################################


class Object(dict):
    """
    A single object in an LHCO event, e.g. a single electron.

    This object inherits the dictionary class - it is itself a dictionary. The
    keys correspond to an object's properties:
    - event
    - type
    - eta
    - phi
    - PT
    - jmass
    - ntrk
    - btag
    - hadem
    """

    def __init__(self, name=None, dictionary=None):
        """
        Initalize a single object, e.g. a single electron.

        Arguments:
        name -- name of object, e.g. electron, muon etc.
        dictionary -- A dictionary, zipped lists etc for a new dictionary
        """

        # Ordinary initialization
        if dictionary: 
            super(self.__class__, self).__init__(dictionary) 
            
        self.name = name  # Save name of object

        # List of object properties in LHCO file, these are row headings in
        # LHCO format, see http://madgraph.phys.ucl.ac.be/Manual/lhco.html
        self._properties = ["event",
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
                            ]

        # List of properties suitable for printing
        self._print_properties = ["eta",
                                  "phi",
                                  "PT",
                                  "jmass",
                                  "ntrk",
                                  "btag",
                                  "hadem"
                                  ]

    def __str__(self):
        """
        String Object, e.g. properties about an electron, into an easy to read
        format.
        """

        # Make table of event
        headings = ["Object"] + self._print_properties
        table = pt(headings)
        table.add_row(self._row())

        return table.get_string()

    def _row(self):
        """
        Make a row for object in an easy to read format.

        This attribute is intended to be semi-private - i.e. you are not
        encouraged to call this function directly.
        """

        row = [self.name]
        for prop in self._print_properties:
            row.append(self[prop])

        return row

    def vector(self):
        """
        Make a four-momentum vector for this Object.

        E.g., an electron's four-momentum from its PT, eta, and phi
        parameters. The four-momentum is a dictionary entry:
        - vector

        This attribute is intended to be semi-private - i.e. you are not
        encouraged to call this function directly.

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

    def LHCO(self, file_name):
        """
        Write object in LHCO format.

        Arguments:
        file_name -- Name of LHCO file to be written
        """
        list_ = []
        for key in self._properties:
            list_.append(repr(self[key]).ljust(10))
        print(*list_, file=open(file_name, "a"))

###############################################################################


class Fourvector(np.ndarray):
    """
    A four-vector, with relevant addition, multiplication etc. operations.

    Builds a four-vector from Cartesian co-ordinates. Defines Minkowski
    product, square, additon of four-vectors.

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

    def __new__(self, v=None):
        """
        Make four-vector from Cartesian co-ordinates.

        Arguments:
        v -- Length 4 list of four-vector in Cartesian co-ordinates
        """
        if v is None:
            v = [0] * 4  # Default is empty four-vector
        elif len(v) != 4:
            raise("Four-vector must be length 4!")
            
        return np.asarray(v).view(self)

    def __init__(self, v=None):
        """
        Initialize four-vector from Cartesian co-ordinates.

        Arguments:
        v -- Length 4 list of four-vector in Cartesian co-ordinates
        """
        self.metric = np.diag([1, -1, -1, -1])  # Define metric

    def __mul__(self, other):
        """
        Multiply four-vectors with Minkowski product, returning a scalar.

        If one entry is in fact a float or an integer, regular multiplication,
        returning a new four-vector.

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
            raise Exception("Unsupported multiplication: " + type(other))

    def __rmul__(self, other):
        """ Four-vector multiplication. See __mul__. """
        return self.__mul__(other)

    def __pow__(self, power):
        """
        Raising a four-vector to a power.

        Only power 2 (Minkowski square) supported.

        >>> x = [1,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p**2)
        -2.0
        """
        if not power == 2:
            raise Exception("Only power 2 is supported: " + power)

        return self.__mul__(self)

    def __str__(self):
        """ String a four-vector for nice printing. """

        headings = ["E", "P_x", "P_y", "P_z"]
        table = pt(headings)
        table.add_row(self.tolist())

        return str(table)

    def __abs__(self):
        """
        Absolute value of a four-vector.

        The square-root of the Minkowski square.

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
        """
        Find angle phi around beam line.

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p.phi())
        0.785398163397
        """
        tan_phi = self[2] / self[1]
        phi = np.arctan(tan_phi)
        return phi

    def PT(self):
        """
        Return PT - transverse magnitude of vector.
        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p.PT())
        1.41421356237
        """
        return (self[1]**2 + self[2]**2)**0.5

    def theta(self):
        """
        Find angle theta between vector and beam line.

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
        """
        Find pseudo-rapidity.

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p.eta())
        0.658478948462
        """
        theta = self.theta()
        eta = -np.log(np.tan(theta/2.))  # Definiton of pseudo-rapidity
        return eta

    def boost(self, beta):
        """
        Boost four-vector into a new refrence-frame.

        Arguments:
        beta -- Numpy array of beta for boost (b1, b2, b3)

        Returns:
        boosted -- Fourvector class, self but boosted by beta

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
        beta_matrix = np.identity(3) + np.outer(beta, beta) * (gamma - 1.) / beta_norm**2

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
        """
        Find gamma for this four-vector:
        
        gamma = p_0 / abs(p)
        
        Returns:
        gamma -- Lorentz factor for this four-vector.
        
        >>> x = [5,1,0,0]
        >>> p = Fourvector(x)
        >>> print(p.gamma())
        1.02062072616

        """
        
        gamma = self[0] / abs(self)  # gamma = E / M
        return gamma

    def beta(self):
        """
        Find beta for this four-vector:
        
        Returns:
        beta -- Beta for this four-vector.
        
        >>> x = [5,1,0,0]
        >>> p = Fourvector(x)
        >>> print(p.beta())
        0.2

        """
        
        beta = (1. - self.gamma()**-2)**0.5
        return beta

    def beta_rest(self):
        """
        Find beta for Lorentz boost to a frame in which this four-vector is at
        rest.

        Returns:
        beta -- Numpy array of beta (b1, b2, b3)

        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p.beta_rest())
        [ 0.2  0.2  0.2]

        """
        beta_norm = self.beta()
        beta = self.unit_vector() * beta_norm

        return beta

    def unit_vector(self):
        """
        Find unit vector in 3-vector direction.
       
        Returns:
        unit -- Unit vector
        
        >>> x = [5,1,1,1]
        >>> p = Fourvector(x)
        >>> print(p.unit_vector())
        [ 0.57735027  0.57735027  0.57735027]
        """
        unit = self[1:] / (self[1]**2 + self[2]**2 + self[3]**2)**0.5
        return unit

###############################################################################


def Fourvector_eta(PT, eta, phi, mass=0.):
    """
    Builds a four-vector from p, eta, phi co-ordinates.

    Convention is that z-direction is the beam line.

    Arguments:
    PT -- transverse momentum
    eta -- pseudo-rapidity, -ln[tan(theta/2)], with theta angle to beam axis
    phi -- azimuthal angle, angle around beam
    mass -- mass of particle

    Returns:
    Fourvector -- Four-vector object

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
    """
    Find the angular separation between two objects.

    Arguments:
    o1 -- An Object, e.g. an electron
    o2 -- Second Object, e.g. jet

    Returns:
    delta_R -- Angular separation between objects

    >>> delta_R_12 = delta_R(events[0]["jet"][1], events[0]["jet"][2])
    >>> print(delta_R_12)
    4.80541371788
    """
    delta_R = ((o1["eta"] - o2["eta"])**2 + (o1["phi"] - o2["phi"])**2)**0.5

    return delta_R

###############################################################################


def comment(x):
    """
    Places # at the beginning of every line in a string or an object to be
    represented as a string.

    Arguments:
    x -- string to be commented

    """
    return "# " + str(x).replace("\n", "\n# ")


###############################################################################


if __name__ == "__main__":
    import doctest
    doctest.testmod(extraglobs={'events': Events("example.lhco")})
