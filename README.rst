LHCO_reader 
***********

============
Introduction
============

:mod:`LHCO_reader` is a Python module for reading `LHCO files <http://madgraph.phys.ucl.ac.be/Manual/lhco.html>`_ from detector simulators such as `PGS <http://www.physics.ucdavis.edu/~conway/research/software/pgs/pgs4-general.htm>`_ into a Python class, with useful functions for implementing an analysis. It can also read ROOT files from `Delphes <https://cp3.irmp.ucl.ac.be/projects/delphes>`_, by immediately converting them to LHCO files. :mod:`LHCO_reader` can calculate :math:`\alpha_T` and razor variables, and :math:`M_{T2}` and :math:`M_{T2}^W` variables are included by linking :mod:`LHCO_reader` with external libraries.

For the full documentation, `read the online docs <http://lhco-reader.readthedocs.org/>`_. For a tutorial and further background information, see the manual at `arXiv:1510.07319 <http://arxiv.org/abs/1510.07319>`_.

Submit any bugs or issues at the `git-hub issues page <https://github.com/innisfree/LHCO_reader/issues>`_.

=========
Reference
=========

If you use :mod:`LHCO_reader`, please cite `arXiv:1510.07319 <http://arxiv.org/abs/1510.07319>`_:: 

    @article{Fowlie:2015dga,
        author         = "Fowlie, Andrew",
        title          = "{LHCO_reader: A new code for reading and analyzing
                        detector-level events stored in LHCO format}",
        year           = "2015",
        eprint         = "1510.07319",
        archivePrefix  = "arXiv",
        primaryClass   = "hep-ph",
        reportNumber   = "COEPP-MN-15-10",
        SLACcitation   = "%%CITATION = ARXIV:1510.07319;%%"
    }
    
============
Installation
============

The module does not require complicated installation. Simply::

    pip install LHCO_reader

or clone the module for the very-latest version::

    git clone https://github.com/innisfree/LHCO_reader.git

or `download it via a web browser <https://github.com/innisfree/LHCO_reader/archive/master.zip>`_.

===========
Quick-start
===========

To load the module :mod:`LHCO_reader` and look at an LHCO file, simply::

    from LHCO_reader import LHCO_reader
    events = LHCO_reader.Events(f_name="example.lhco")
    print events

The :class:`Events` object in the above code is a list-like object. Cuts can be implemented with lambda-functions, e.g. to cut events with one tau-lepton::

    tau = lambda event: event.number()["tau"] == 1
    events.cut(tau)
     
To test the module::

    python LHCO_reader.py -v

The module requires some common modules that you might need to install separately, the most obscure of which is :mod:`prettytable`, see  `here for installation <https://code.google.com/p/prettytable/wiki/Installation>`_.

===================
Structure of events
===================

The code is object-oriented. A LHCO file is parsed into several objects. 
The :class:`Events` object is structured as follows:

- :class:`Events` - A list of all events in the LHCO file

- :class:`Events[0]` - The zeroth event in the LHCO file. The :class:`Events` can be looped with e.g.:

.. code-block:: python

    for event in events:
      ... scrutinize an event ...
 
but beware that altering list-type objects in a loop can be problematic. The best way to cut :class:`Events` is with the :func:`Events.cut` function.
    
- :class:`Events[0]["electron"]` - A list of all electrons in the zeroth event in the LHCO file. For ordinary LHCO files, the possible keys are :literal:`electron`, :literal:`muon`, :literal:`tau`, :literal:`jet`, :literal:`MET` and :literal:`photon`.

- :class:`Events[0]["electron"][0]` - The zeroth electron in the zeroth event in the LHCO file.
  
- :class:`Events[0]["electron"][0]["PT"]` - The transverse momentum of the zeroth electron in the zeroth event in the LHCO file. The other possible keys are :literal:`event,` :literal:`type`, :literal:`eta`, :literal:`phi`, :literal:`PT`, :literal:`jmass`, :literal:`ntrk`, :literal:`btag` and :literal:`hadem`.
 
There are many useful functions, including printing in LHCO format (:func:`LHCO`), plotting (:func:`plot`), sorting (:func:`order`) and cutting events (:func:`cut`), manipulating four-momenta with boosts (:func:`vector`), counting the numbers of types of object in an event (:func:`number`), angular separation (:func:`delta_R`), that should make implementing an analysis easy.

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

Kinematic variables
===================

Complicated kinematic variables could be included from the Oxbridge kinetics
library.

>>> object_1 = events[0]["jet"][0]
>>> object_2 = events[0]["jet"][1]
>>> MET = events[0]["MET"][0]
>>> from oxbridge_kinetics import MT2
>>> MT2(object_1, object_2, MET)
53.305931964300186

====
ROOT
====

ROOT files can be converted into LHCO files with :mod:`root2lhco` in `Delphes <https://cp3.irmp.ucl.ac.be/projects/delphes>`_, which can be linked with and called from within :mod:`LHCO_reader` via :mod:`LHCO_converter`, i.e. you can load a ROOT file, which will be immediately converted into an LHCO file and parsed. If you wish to use ROOT files::

    export DELPHES=MY/PATH/TO/DELPHES   
