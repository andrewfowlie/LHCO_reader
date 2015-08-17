LHCO_reader 
===========

LHCO_reader is a Python module for reading [LHCO files](http://madgraph.phys.ucl.ac.be/Manual/lhco.html) from detector simulators such as [PGS](http://www.physics.ucdavis.edu/~conway/research/software/pgs/pgs4-general.htm) into a Python class, with useful functions for implementing an analysis. The module does not require installation. 

ROOT files can be converted into LHCO files with `root2lhco` in [Delphes](https://cp3.irmp.ucl.ac.be/projects/delphes).

To load the module and look at an LHCO file, simply

    import LHCO_reader
    events = LHCO_reader.Events(f_name="example.lhco")
    print events
    
To test the module,

    python LHC_reader.py -v

The modue requires some common modules that you might need to install separately, the most obscure of which is [`prettytable`](https://code.google.com/p/prettytable/wiki/Installation).

The `events` object in the above code is an (list) object. Cuts can be implemented with lambda-functions, e.g. to cut events with one tau-lepton:

    tau = lambda event: event.number()["tau"] == 1
    events.cut(tau)
   
The `events` object is structured as follows:

`events` -- A list of all events in the LHCO file

`events[0]` -- The zeroth event in the LHCO file. The events can be looped with e.g.

    for event in events:
      ... scrutinize an event ...
 
but beware that altering list-type objects in a loop can be problematic.
    
`events[0]["electron"]` -- A list of all electrons in the zeroth event in the LHCO file. For ordinary LHCO files, the possible keys are electron, muon, tau, jet, MET and photon.

`events[0]["electron"][0]` -- The zeroth electron in the zeroth event in the LHCO file.
  
`events[0]["electron"][0]["PT"]` -- The transverse momentum of the zeroth electron in the zeroth event in the LHCO file. The other possible keys are event, type, eta, phi, PT, jmass, ntrk, btag and hadem.
 
There are many useful functions, including printing in LHCO format, plotting, sorting and cutting events, manipulating four-momenta with boosts, counting the numbers of types of object in an event, angular separation etc, that should make implementing an analysis easy.
