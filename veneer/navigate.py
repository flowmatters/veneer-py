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
            super(Queryable,self).__setattr__(r,self._child_(r))

    def _run_script(self,script):
        return self._v.model._safe_run('%s\n%s'%(self._v.model._init_script(),script))

    def __call__(self,*args,**kwargs):
        return self._v.model.call(self._path+str(tuple(args)))

    def __repr__(self):
        return str(self._eval_())

    def __dir__(self):
        res = [e['Value'] for e in self._run_script('dir(%s)'%(self._path))['Response']['Value']]
        self._initialise_children_(res)
        return res
    
    def __getattr__(self,attrname):
        return self._child_(attrname)

    def __getitem__(self,ix):
        return self._child_idx_(ix)

    def __setattr__(self,a,v):
        if a.startswith('_'):
            return super(Queryable,self).__setattr__(a,v)

        v = self._double_quote_(v)
            
        if not self._v.model.set('%s.%s'%(self._path,a),v):
            raise Exception("Couldn't set property")
