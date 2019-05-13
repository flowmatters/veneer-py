import pandas as pd
import re
import inspect
import warnings

CANARY_METHODS = [
    '_ipython_canary_method_should_not_exist_',
    '_ipython_display_',
    '_repr_javascript_',
    '_repr_jpeg_',
    '_repr_latex_',
    '_repr_markdown_',
    '_repr_pdf_',
    '_repr_json_',
    '_repr_html_',
    '_repr_svg_',
    '_repr_png_'
]

try:
    from collections import UserDict
except ImportError:
    from UserDict import UserDict


def deprecate_async(cls):
    """Decorator to warn of deprecated `async` argument.

    `async` has become a reserved keyword as of Python 3.7
    """
    class DepAsync(object):
        def __init__(self, *args, **kwargs):
            self._wrapped = cls(*args, **kwargs)

        def __getattr__(self, name):
            attr = getattr(self._wrapped, name)
            if not inspect.ismethod(attr):
                return attr

            def wrapped(*args, **kwargs):
                if 'async' in kwargs:
                    warnings.warn("Use of deprecated `async` argument. \
                                   Use `run_async` instead",
                                  DeprecationWarning)
                    use_async = kwargs.pop('async', False)

                    # in case they use both, prefer value for run_async
                    if 'run_async' in kwargs:
                        warnings.warn("Both `async` and `run_async` arguments \
                                       used. In future, just use `run_async`.")
                    else:
                        kwargs['run_async'] = use_async

                return attr(*args, **kwargs)

            return wrapped

    return DepAsync


def read_veneer_csv(text):
    return text


def objdict(orig):
    return UserDict(orig)


# class objdict(dict):
#    def __init__(self,initial={}):
#        for k,v in initial.items():
#            self[k] = v
#
#    def __getattr__(self, name):
#        if name in self:
#            return self[name]
#        else:
#            raise AttributeError("No such attribute: " + name)
#
#    def __setattr__(self, name, value):
#        self[name] = value
#
#    def __delattr__(self, name):
#        if name in self:
#            del self[name]
#        else:
#            raise AttributeError("No such attribute: " + name)

class GroupedDictionary(UserDict):
    def __init__(self, initial={}):
        super(GroupedDictionary, self).__init__()
        self.update(initial)

    def count(self):
        return {k: len(self[k]) for k in self}


class SearchableList(object):
    '''
    SearchableList of objects

    Use
      * find_by_X(Y) to find all entries in the list where property X is equal to Y
      * group_by_X() to return a Python dictionary with keys being the unique values of property X and entries
                     being a list of original list entries with matching X values
      * _unique_values(X) return a set of unique values of property X
      * _all_values(X) return a list of all values of property X (a simple select)
      * _select(['X','Y']) to select particular properties
    For example:

    v = Veneer()
    network = v.network()
    the_list = network['features']  # A searchable list of network features
    links = the_list.find_by_feature_type('link')   # All features have a 'feature_type' property

    grouped = the_list.group_by_feature_type()
    grouped['link'] is a list of links
    grouped['node'] is a list of nodes
    grouped['catchment'] is a list of catchments
    '''

    def __init__(self, the_list, nested=[]):
        self._list = the_list
        self._nested = nested

    def __repr__(self):
        return self._list.__repr__()

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __iter__(self):
        return self._list.__iter__()

    def __reversed__(self):
        return SearchableList(reversed(self._list))

    def __contains__(self, item):
        return self._list.__contains__(item)

    def _match(self, entry, test, op):
        if op == '=':
            return entry == test
        if op == '!=':
            return entry != test
        if op == '>':
            return entry > test
        if op == '>=':
            return entry >= test
        if op == '<':
            return entry > test
        if op == '<=':
            return entry <= test
        if op == 'LIKE':
            return entry.find(test) >= 0
        if op == 'STARTS':
            return entry.startswith(test)
        if op == 'ENDS':
            return entry.endswith(test)
        raise Exception('Unknown operation')

    def _search_all(self, key, val, entry, op):
        if (key in entry) and self._match(entry[key], val, op):
            return True
        for nested in self._nested:
            if not nested in entry:
                continue
            if not key in entry[nested]:
                continue
            if self._match(entry[nested][key], val, op):
                return True
        return False

    def _nested_retrieve(self, key, entry):
        if (key in entry):
            return entry[key]
        for nested in self._nested:
            if not nested in entry:
                continue
            if key in entry[nested]:
                return entry[nested][key]
        return None

    def _unique_values(self, key):
        return set(self._all_values(key))

    def _all_values(self, key):
        return [self._nested_retrieve(key, e) for e in self._list]

    def _select(self, keys, transforms={}):
        result = [{k: self._nested_retrieve(k, e) for k in keys} for e in self]

        for key, fn in transforms.items():
            for r, e in zip(result, self):
                r[key] = fn(e)

        if len(result) == 0:
            SearchableList([])
        elif len(result[0]) == 1:
            key = list(result[0].keys())[0]
            return [r[key] for r in result]

        return SearchableList(result)

    def __getattr__(self, name):
        FIND_PREFIX = 'find_by_'
        if name.startswith(FIND_PREFIX):
            field_name = name[len(FIND_PREFIX):]
            return lambda x, op='=': SearchableList(list(filter(lambda y: self._search_all(field_name, x, y, op), self._list)), self._nested)

        FIND_ONE_PREFIX = 'find_one_by_'
        if name.startswith(FIND_ONE_PREFIX):
            field_name = name[len(FIND_ONE_PREFIX):]
            return lambda x, op='=': list(filter(lambda y: self._search_all(field_name, x, y, op), self._list))[0]

        GROUP_PREFIX = 'group_by_'
        if name.startswith(GROUP_PREFIX):
            field_name = name[len(GROUP_PREFIX):]
            return lambda: GroupedDictionary({k: self.__getattr__(FIND_PREFIX + field_name)(k) for k in self._unique_values(field_name)})

        find_suffixes = {
            '_like': 'LIKE',
            '_startswith': 'STARTS',
            '_endswith': 'ENDS'
        }
        for suffix, op in find_suffixes.items():
            if name.endswith(suffix):
                field_name = name[0:-len(suffix)]
                return lambda x: SearchableList(list(filter(lambda y: self._search_all(field_name, x, y, op), self._list)), self._nested)
        raise AttributeError(name + ' not allowed')

    def as_dataframe(self):
        return pd.DataFrame(self._list)


class DeferredCall(object):
    def __init__(self, parameter, delimiter):
        if parameter:
            self.call_tree = [parameter]
        else:
            self.call_tree = []
        self.pargs = []
        self.kwargs = {}
        self.delimiter = delimiter
        self.cal_params = []
        self._is_call = False
        self.next = None

    def __getattr__(self, attrname):
        if not attrname in CANARY_METHODS:
            self.call_tree.append(attrname)
        return self

    def __getitem__(self, ix):
        i = len(self.call_tree) - 1
        if i < 0:
            self.call_tree.append('')
        self.call_tree[i] += '[%s]' % str(ix)
        return self

    def __call__(self, *pargs, **kwargs):
        self.pargs = pargs
        self.kwargs = kwargs
        self.cal_params = [p for p in (
            list(self.pargs) + list(self.kwargs.values())) if self.is_cal_param(p)]
        self._is_call = True
        self.next = DeferredCall(None, self.delimiter)
        return self.next

    def is_cal_param(self, val):
        return isinstance(val, str) and (val[0] == self.delimiter) and (val[-1] == self.delimiter)

    def to_arg(self, val):
        if isinstance(val, str) and not self.is_cal_param(val):
            return "'%s'" % val
        return val

    def argstring(self):
        pstring = ','.join([self.to_arg(v) for v in self.pargs])
        kwstring = ','.join(['%s=%s' % (k, str(self.to_arg(v)))
                             for k, v in self.kwargs.items()])
        if len(pstring) and len(kwstring):
            return ','.join([pstring, kwstring])
        else:
            return pstring + kwstring

    def __str__(self):
        result = '.'.join(self.call_tree)
        if self._is_call:
            result += '(' + self.argstring() + ')'
        if self.next:
            next_text = str(self.next)
            if next_text:
                if next_text[0] == '[':
                    result += next_text
                else:
                    result += '.' + next_text
        return result


class DeferredActionCollection(object):
    def __init__(self, delimiter, instruction_prefix):
        self.instructions = []
        self.instruction_prefix = instruction_prefix
        self.delimiter = delimiter

    def __getattr__(self, attrname):
        if attrname in ['_getAttributeNames', 'trait_names']:
            raise Exception('Not implemented')
        self.instructions.append(DeferredCall(attrname, self.delimiter))
        return self.instructions[-1]

    def script_line(self, instruction, transform=None):
        if transform is None:
            return self.instruction_prefix + str(instruction)
        else:
            return transform(instruction)

    def script(self, transform=None):
        return '\n'.join([self.script_line(i, transform) for i in self.instructions])

    def eval_script(self, substitutions={}, global_variables={}):
        script = self.script()
        for key, val in substitutions.items():
            if key[0] != self.delimiter:
                key = '%s%s%s' % (self.delimiter, key, self.delimiter)
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            script = pattern.sub(str(val), script)
        exec(script, global_variables)
        # for line in script.splitlines():
        #   eval(line,global_variables)


def _stringToList(string_or_list):
    if isinstance(string_or_list, str):
        return [string_or_list]
    return string_or_list


def _variable_safe_name(name):
    SUBSTS = '- \t'
    STRIP = '#@!$%*&()[]{}'
    for c in SUBSTS:
        name = name.replace(c, '_')
    for c in STRIP:
        name = name.replace(c, '')
    name = re.sub('__+', '_', name)
    name = name.strip('_')
    return name


def _safe_filename(fn):
    return fn.replace('\\', '/')

def split_network(network):
    '''
    Takes a Source network (GeoDataFrame) and returns separate GeoDataFrames for 

    * links,
    * nodes, and
    * catchments

    in that order
    '''
    links = network[network.feature_type=='link']
    nodes = network[network.feature_type=='node']
    catchments = network[network.feature_type=='catchment']
    return links,nodes,catchments

