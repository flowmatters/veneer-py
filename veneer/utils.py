import pandas as pd
import re

try:
    from collections import UserDict
except ImportError:
    from UserDict import UserDict


def read_veneer_csv(text):
    return text

def objdict(orig):
    return UserDict(orig)


#class objdict(dict):
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
    def __init__(self,initial={}):
        super(GroupedDictionary,self).__init__()
        self.update(initial)

    def count(self):
        return {k:len(self[k]) for k in self}

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
    def __init__(self,the_list,nested=[]):
        self._list = the_list
        self._nested = nested

    def __repr__(self):
        return self._list.__repr__()

    def __len__(self):
        return len(self._list)

    def __getitem__(self,i):
        return self._list[i]

    def __iter__(self):
        return self._list.__iter__()

    def __reversed__(self):
        return SearchableList(reversed(self._list))

    def __contains__(self,item):
        return self._list.__contains__(item)

    def _match(self,entry,test,op):
        if op=='=':
            return entry==test
        if op=='!=':
            return entry!=test
        if op=='>':
            return entry>test
        if op=='>=':
            return entry>=test
        if op=='<':
            return entry>test
        if op=='<=':
            return entry<=test
        if op=='LIKE':
            return entry.find(test)>=0
        if op=='STARTS':
            return entry.startswith(test)
        if op=='ENDS':
            return entry.endswith(test)
        raise Exception('Unknown operation')

    def _search_all(self,key,val,entry,op):
        if (key in entry) and self._match(entry[key],val,op): return True
        for nested in self._nested:
            if not nested in entry: continue
            if not key in entry[nested]: continue
            if self._match(entry[nested][key],val,op): return True
        return False

    def _nested_retrieve(self,key,entry):
        if (key in entry): return entry[key]
        for nested in self._nested:
            if not nested in entry: continue
            if key in entry[nested]: return entry[nested][key]
        return None

    def _unique_values(self,key):
        return set(self._all_values(key))

    def _all_values(self,key):
        return [self._nested_retrieve(key,e) for e in self._list]

    def _select(self,keys,transforms={}):
        result = [{k:self._nested_retrieve(k,e) for k in keys} for e in self]

        for key,fn in transforms.items():
            for r,e in zip(result,self):
                r[key] = fn(e)

        if len(result)==0: SearchableList([])
        elif len(result[0])==1:
            key = list(result[0].keys())[0]
            return [r[key] for r in result]

        return SearchableList(result)

    def __getattr__(self,name):
        FIND_PREFIX='find_by_'
        if name.startswith(FIND_PREFIX):
            field_name = name[len(FIND_PREFIX):]
            return lambda x,op='=': SearchableList(list(filter(lambda y: self._search_all(field_name,x,y,op),self._list)),self._nested)

        FIND_ONE_PREFIX='find_one_by_'
        if name.startswith(FIND_ONE_PREFIX):
            field_name = name[len(FIND_ONE_PREFIX):]
            return lambda x,op='=': list(filter(lambda y: self._search_all(field_name,x,y,op),self._list))[0]

        GROUP_PREFIX='group_by_'
        if name.startswith(GROUP_PREFIX):
            field_name = name[len(GROUP_PREFIX):]
            return lambda: GroupedDictionary({k:self.__getattr__(FIND_PREFIX+field_name)(k) for k in self._unique_values(field_name)})

        find_suffixes = {
            '_like':'LIKE',
            '_startswith':'STARTS',
            '_endswith':'ENDS'
        }
        for suffix,op in find_suffixes.items():
            if name.endswith(suffix):
                field_name = name[0:-len(suffix)]
                return lambda x: SearchableList(list(filter(lambda y: self._search_all(field_name,x,y,op),self._list)),self._nested)
        raise AttributeError(name + ' not allowed')


    def as_dataframe(self):
        return pd.DataFrame(self._list)

def _stringToList(string_or_list):
    if isinstance(string_or_list,str):
        return [string_or_list]
    return string_or_list

def _variable_safe_name(name):
    SUBSTS='- \t'
    STRIP='#@!$%*&()[]{}'
    for c in SUBSTS:
        name = name.replace(c,'_')
    for c in STRIP:
        name = name.replace(c,'')
    name = re.sub('__+','_',name)
    name = name.strip('_')
    return name

def _safe_filename(fn):
    return fn.replace('\\','/')
