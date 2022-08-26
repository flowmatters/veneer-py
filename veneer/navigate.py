'''
Prototype functionality for interacting with the Source model directly, including tab-completion in IPython/Jupyter. Eg

v = veneer.Veneer()
scenario = Queryable(v)
scenario.Name = 'New Scenario Name'
'''

class Queryable(object):
    def __init__(self,v,path='scenario',namespace=None):
        self._v = v
        self._path = path
        self._init = False
        self._ns = namespace

    def _eval_(self):
        return self._v.model.get(self._path,namespace=self._ns)

    def _child_(self,path):
        val = Queryable(self._v,'%s.%s'%(self._path,path),namespace=self._ns)
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
        return Queryable(self._v,'%s[%s]'%(self._path,str(ix)),namespace=self._ns)

    def _initialise_children_(self,entries):
        if self._init: return
        self._init = True
        for r in entries:
            if r[:2]=='__': continue
            super(Queryable,self).__setattr__(r,self._child_(r))

    def _run_script(self,script):
        return self._v.model._safe_run('%s\n%s'%(self._v.model._init_script(self._ns),script))

    def __call__(self,*args,**kwargs):
        arguments = [_format_for_script(a) for a in args]
        definitions = 'clr\n'+'\n'.join([d for d,_ in arguments])

        args = [a for _,a in arguments]
        args = ','.join(args)
        return self._v.model.call(f'{self._path}({args})',namespace=definitions)

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

    def __int__(self):
        return int(self._eval_())
    
    def __float__(self):
        return float(self._eval_())


def _format_for_script(o):
    import inspect
    if o.__class__==inspect.getsource.__class__:
        code = inspect.getsource(o).strip()
        defn = ''
        lines = code.splitlines()
        if '= lambda' in lines[0]:
            defn = code
            fn_name = lines[0][:lines[0].index('=')].strip()
            code = fn_name
        elif lines[0].startswith('def'):
            defn = code
            fn_name = lines[0].replace('def ','')
            fn_name = fn_name[:fn_name.index('(')]
            code = fn_name
        else:
            assert len(lines)==1
            code = code[code.index('lambda'):]
            count_open = code.count('(')
            count_closed = code.count(')')
            while count_closed > count_open:
                code = code.strip()
                code = code[:-1]
                count_closed = code.count(')')
        return defn,code
    return '',o

