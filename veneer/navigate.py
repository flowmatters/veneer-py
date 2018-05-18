'''
Prototype functionality for interacting with the Source model directly, including tab-completion in IPython/Jupyter. Eg

v = veneer.Veneer()
scenario = Queryable(v)
scenario.Name = 'New Scenario Name'
'''

class Queryable(object):
    def __init__(self,v,path='scenario'):
        self._v = v
        self._path = path
        self._init = False

    def _eval_(self):
        return self._v.model.get(self._path)

    def _child_(self,path):
        val = Queryable(self._v,'%s.%s'%(self._path,path))
        #prop = CustomProperty(path,val)
        return val

    def _double_quote_(self,maybe_string):
        v = maybe_string
        if not isinstance(v,str):
            return v
        if not "'" in v:
            return "'%s'"%v
        if not '"' in v:
            return '"%s"'%v

        v = v.replace('"','\\"')
        return '"%s"'%v

    def _child_idx_(self,ix):
        return Queryable(self._v,'%s[%s]'%(self._path,str(ix)))

    def _initialise_children_(self,entries):
        if self._init: return
        self._init = True
        for r in entries:
            if r[:2]=='__': continue
            setattr(self,r,self._child_(r))

    def __repr__(self):
        return str(self._eval_())

    def __dir__(self):
        res = [e['Value'] for e in self._v.model.run_script('dir(%s)'%self._path)['Response']['Value']]
        self._initialise_children_(res)
        return res
    
    def __getattr__(self,attrname):
        return self._child_(attrname)
#         if not attrname in CANARY_METHODS:
#             self.call_tree.append(attrname)
#         return self

    def __getitem__(self,ix):
        return self._child_idx_(ix)
#         i = len(self.call_tree)-1
#         if i<0:
#           self.call_tree.append('')
#         self.call_tree[i] += '[%s]'%str(ix)
#         return self

    def __setattr__(self,a,v):
        if a.startswith('_'):
            return super(Queryable,self).__setattr__(a,v)

        v = self._double_quote_(v)
            
        self._v.model.set('%s.%s'%(self._path,a),v)