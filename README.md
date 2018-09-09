# veneer-py
Python module to support scripting eWater Source models through the Veneer (RESTful HTTP) plugin

## Installation

You'll need Python 3, a compatible version of eWater Source along with the [Veneer plugin](https://github.com/flowmatters/veneer) and veneer-py.

I expect most users of veneer-py will install [Anaconda Python](https://www.continuum.io/downloads). Anaconda will provide most of the analytics libraries you're likely to want, along with the Jupyter Notebook system. Install the most recent version of Anaconda Python with Python 3 (NOT Python 2).

Instructions for installing Veneer can be found on [its homepage](https://github.com/flowmatters/veneer). Download  the most recent [Veneer release](https://github.com/flowmatters/Veneer/releases) that is compatible with your version of eWater Source. Note that certain veneer-py features may not work with older versions of Source/Veneer.

veneer-py can be installed using `pip` from an Anaconda command prompt:

```
pip install https://github.com/flowmatters/veneer-py/archive/master.zip
```

At this stage we haven't tagged releases so you just install from the latest version.

To upgrade, uninstall the one you've got, then install again

```
pip uninstall veneer-py
pip install https://github.com/flowmatters/veneer-py/archive/master.zip
```

Alternatively, clone the git repository and do a develop install to allow you to easily modify the veneer-py code as you use it.

```
python setup.py develop
```

## Getting started

1. Install the Veneer plugin for Source as per its [instructions](https://github.com/flowmatters/veneer).
2. Start Source, load a project and then start the Veneer service from within Source.
3. Within Python (eg within a notebook), initialise a Veneer client object and run a query. For example

```python
from veneer import Veneer
v = Veneer()    # uses default port number of 9876
# Alternatively, for a different port
# v = Veneer(port=9876)

network = v.network()   # Returns a GeoJSON coverage representing the Source network
nodes = network['features'].find_by_feature_type('node')
node_names = nodes._unique_values('name')
print(node_names)
```

## Exploring the system

Most of the key functions of the Veneer object have docstrings and you are encouraged to explore these. Using IPython or the Jupyter Notebook gives you tab key completion, so, for example, you can type

```
v.r
```

Hit `tab` and see a list of methods starting with `r`.

To get help on a specific command, put a `?` after the full method name, such as

```
v.retrieve_multiple_time_series?
```

## Documentation and Training

Reference docs are at [https://flowmatters.github.io/veneer-py](https://flowmatters.github.io/veneer-py).

Training notebooks and sample data are at [doc/training](doc/training)

## Contributions

... Are most welcome... We'll formalise contributor guidelines in the future.
