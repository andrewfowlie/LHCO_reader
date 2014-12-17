LHCO_reader ([Full documentation](http://innisfree.github.io/))
===========

A Python module for reading LHCO files from detector simulators such as PGS into a Python class, with useful functions for implementing an analysis.

The module does not require installation. 

To load the module and look at an LHCO file, simply

    import LHCO_reader
    events = LHCO_reader.Events(f_name="example.lhco")
    print events

The modue requires some common modules that you might need to install separately, the most obscure of which is [`prettytable`](https://code.google.com/p/prettytable/wiki/Installation).

The `events` variable is an (list) object, structured as follows:

`events` -- A list of all events in the LHCO file

`events[0]` -- The zeroth event in the LHCO file. The events can be looped with

    for event in events:
      ... scrutinize an event ...
    
`events[0]["electron"]` -- A list of all electrons in the zeroth event in the LHCO file.

`events[0]["electron"][0]` -- The zeroth electron in the zeroth event in the LHCO file.
  
 `events[0]["electron"][0]["PT"]` -- The transverse momentum of the zeroth electron in the zeroth event in the LHCO file. 
 
There are many useufl functions, including plotting, sorting, four-momenta, counting the numbers of types of object in an event, angular separation etc, that should make implementing an analysis easy - see the [full documentation](http://innisfree.github.io/).
