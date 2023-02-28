import numpy as np
import pandas as pd
import math
import os
import re

from string import Template
from collections import OrderedDict
from subprocess import Popen,PIPE
from shutil import copyfile

from veneer import stats
from .pest_runtime import CONNECTION_FN
from .manage import kill_all_on_exit, kill_all_now, BulkVeneer
from .utils import DeferredActionCollection

dict = OrderedDict

FAILED_RUN_OBJ_FN= 1000000.0
DEFAULT_PEST_DELIMITER='$'

#
PTF_SINGLE_THREADED_INSTRUCTION_PREFIX='v.'
PTF_DATA_IO_INSTRUCTION_PREFIX='pd.'
LOG_FILE='detailed_log.csv'

PTF_PREFIX='''ptf $
LOG_FILE='detailed_log.csv'
import os
import math
from veneer.pest_runtime import *
from veneer import Veneer
from veneer.stats import * 
import pandas as pd
import numpy as np
from veneer import general
general.PRINT_URLS=False
veneer_port=find_port()
v = Veneer(port=veneer_port)
detailed_log = %s
run_details=[]
'''

PTF_RUN='''
# Run Model
v.drop_all_runs()
v.run_model(**%s)

observed_ts={}
pest_observations=[]

# Get results
run_results = v.retrieve_run()

if 'RunLog' in run_results:
  print(os.linesep.join(run_results['RunLog']))
else:
  print('No run log. Consider upgrading Veneer plugin')

failed = False
if 'Status' in run_results:
  if run_results['Status'] != 'RunSuccess':
    print('Run failed')
    with open('failed_run_log.txt','w') as fp:
      fp.write(os.linesep.join(run_results['RunLog']))
    failed = True

# Compute stats
'''

PTF_ASSESS='''

# Write summary results
print(pest_observations)
write_outputs(pest_observations,'%s')
write_outputs(pest_observations,'__outputs_to_keep.txt')
'''

WRITE_DETAILED_LOG='''
run_details += pest_observations

if not os.path.exists(LOG_FILE):
  f = open(LOG_FILE,'w')
  f.write(','.join([n for n,_ in run_details])+'\\n')
else:
  f = open(LOG_FILE,'a')

f.write(','.join([str(v) for _,v in run_details])+'\\n')
'''

PIF_PREFIX='pif $'

PCF=Template('''pcf
* control data
restart estimation
$NPAR $NOBS $NPARGP $NPRIOR $NOBSGP $MAXCOMPDIM $DERZEROLIM
$NTPLFLE $NINSFLE $PRECIS $DPOINT $NUMCOM $JACFILE $MESSFILE $OBSREREF
$RLAMBDA1 $RLAMFAC $PHIRATSUF $PHIREDLAM $NUMLAM $JACUPDATE $LAMFORGIVE $DERFORGIVE
$RELPARMAX $FACPARMAX $FACORIG $IBOUNDSTICK $UPVECBEND $ABSPARMAX
$PHIREDSWH $NOPTSWITCH $SPLITSWH $DOAUI $DOSENREUSE $BOUNDSCALE
$NOPTMAX $PHIREDSTP $NPHISTP $NPHINORED $RELPARSTP $NRELPAR $PHISTOPTHRESH $LASTRUN $PHIABANDON
$ICOV $ICOR $IEIG $IRES $JCOSAVE $VERBOSEREC $JCOSAVEITN $REISAVEITN $PARSAVEITN $PARSAVERUN
* singular value decomposition
$SVDMODE
$MAXSING $EIGTHRESH
$EIGWRITE
* parameter groups
$PARAMETER_GROUP_LINES
* parameter data
$PARAMETER_LINES $TIED_PARAMETER_LINES
* observation groups
$OBSERVATION_GROUP_LINES
* observation data
$OBSERVATION_LINES
* model command line
python $RUNNER_FN
* model input/output
$TEMPLATE_FILE_LINES
$INSTRUCTION_FILE_LINES
''')

PRF=Template('''prf
$NSLAVE $IFLETYP $WAIT $PARLAM $RUNREPEAT run_slow_fac=$RUN_SLOW_FAC
$SLAVE_LINES
$SLAVE_RUNTIME_LINES
''')

DEFAULT_PG='default_pg'
DEFAULT_OG='default_og'

DECLARE_PARAM=Template('$PARNME $PARTRANS $PARCHGLIM $PARVAL1 $PARLBND $PARUBND $PARGP $SCALE $OFFSET $DERCOM')
CONTROL_DEFAULTS = OrderedDict([('NPAR',None),('NOBS',None),('NPARGP',1),('NPRIOR',0),('NOBSGP',1),('MAXCOMPDIM',''),('DERZEROLIM',''),
  ('NTPLFLE',None),('NINSFLE',None),('PRECIS','double'),('DPOINT','point'),('NUMCOM',''),('JACFILE',''),('MESSFILE',''),('OBSREREF',''),
  ('RLAMBDA1',10.0),('RLAMFAC',2.0),('PHIRATSUF',0.3),('PHIREDLAM',0.03),('NUMLAM',8),('JACUPDATE',''),('LAMFORGIVE',''),('DERFORGIVE',''),
  ('RELPARMAX',10.0),('FACPARMAX',10.0),('FACORIG',0.001),('IBOUNDSTICK',''),('UPVECBEND',''),('ABSPARMAX',''),
  ('PHIREDSWH',0.1),('NOPTSWITCH',''),('SPLITSWH',''),('DOAUI',''),('DOSENREUSE',''),('BOUNDSCALE',''),
  ('NOPTMAX',50),('PHIREDSTP',0.005),('NPHISTP',4),('NPHINORED',4),('RELPARSTP',0.005),('NRELPAR',4),('PHISTOPTHRESH',''),('LASTRUN',''),('PHIABANDON',''),
  ('ICOV',1),('ICOR',1),('IEIG',1),('IRES',''),('JCOSAVE',''),('VERBOSEREC',''),('JCOSAVEITN',''),('REISAVEITN',''),('PARSAVEITN',''),('PARSAVERUN','')])

SVD_DEFAULTS=OrderedDict([('SVDMODE',1),('MAXSING',None),('EIGTHRESH',5e-7),('EIGWRITE',0)])

PARA_GROUP_DEFAULTS=OrderedDict([('PARGPNME',None),('INCTYP','relative'),('DERINC',0.001),('DERINCLB',0.0001),('FORCEN','switch'),('DERINCMUL',1.5),('DERMTHD','parabolic'),('SPLITTHRESH',''),('SPLITRELDIFF',''),('SPLITACTION','')])

PARA_DEFAULTS=OrderedDict([('PARNME',None),('PARTRANS',"none"),('PARCHGLIM','factor'),('PARVAL1',None),('PARLBND',None),('PARUBND',None),('PARGP',DEFAULT_PG),
  ('SCALE',1.0),('OFFSET',0.0),('DERCOM',1)])

TIED_PARA_DEFAULTS=OrderedDict([('PARNME',None),('PARTIED',None)])

OBS_GROUP_DEFAULTS=OrderedDict([('OBGNME',None),('GTARG',''),('COVFLE','')])

OBS_DEFAULTS=OrderedDict([('OBSNME',None),('OBSVAL',None),('WEIGHT',1.0),('OBGNME',DEFAULT_OG)])

TEMPLATE_DEFAULTS=OrderedDict([('TEMPFLE',None),('INFLE',None)])

INSTRUCTION_DEFAULTS=OrderedDict([('INSFLE',None),('OUTFLE',None)])

PRF_DEFAULTS=OrderedDict([('NSLAVE',None),('IFLETYP',0),('WAIT',0.5),('PARLAM',1),('RUNREPEAT',0),('RUN_SLOW_FAC',1.5)])
SLAVE_DEFAULTS=OrderedDict([('SLAVNAME',None),('SLAVDIR',None),('SLAVEGROUP','')])

def validate_dict(the_dict):
  errors = []
  for k,v in the_dict.items():
    if v is None:
      errors.append(k)

  if len(errors):
    raise Exception('Missing required value for %s in %s'%(str(errors),str(the_dict)))

def dec_to_str(val):
  if val is None:
    return ''

  if isinstance(val,float):
    txt = '%0.4f'%val
    txt = txt.rstrip('0')
    if txt.endswith('.'):
      txt +='0'
    return txt

  return str(val)

class ConfigItemCollection(object):
  def __init__(self,template):
    self.template = template
    self.items = []

  def __len__(self):
    return len(self.items)

  def add(self,*pargs,**kwargs):
    options = self.template.copy()
    p_keys = list(options.keys())[:len(pargs)]
    for k,v in zip(p_keys,pargs):
      options[k] = v

    for k,v in kwargs.items():
      options[k.upper()] = v

    self.items.append(options)

  def validate(self):
    for item in self.items:
      validate_dict(item)

  def declarations(self):
    self.validate()
    return '\n'.join([' '.join([dec_to_str(v) for v in entry.values()]) for entry in self.items])

class PestParameter(object):
  def __init__(self,parnme,**kwargs):
    self.parnme = parnme
    self.options = kwargs

  def declaration(self):
    full_options = PARA_DEFAULTS.copy()
    full_options['PARNME']=self.parnme
    for k,v in self.options.items():
      full_options[k.upper()] = v
    return DECLARE_PARAM.substitute(full_options)

class CalibrationParameters(DeferredActionCollection):
  def __init__(self,delimiter=DEFAULT_PEST_DELIMITER,
         instruction_prefix=PTF_SINGLE_THREADED_INSTRUCTION_PREFIX,
         detailed_log=True):
    super(CalibrationParameters,self).__init__(delimiter,instruction_prefix)
    self.params = []
    self.detailed_log = detailed_log

  def __len__(self):
    return len(self.params)

  def describe(self,parnme,parval1,parlbnd,parubnd,**kwargs):
    self.params.append(PestParameter(parnme.replace(self.delimiter,''),parval1=parval1,parlbnd=parlbnd,parubnd=parubnd,**kwargs))

  def referenced_parameters(self):
    return [p for sublist in self.instructions for p in sublist.cal_params]

  def declarations(self):
    return '\n'.join([p.declaration() for p in self.params])

  def log_script(self):
    return '\n'.join(['run_details.append(("%s",%s%s%s))'%(p.parnme,self.delimiter,p.parnme,self.delimiter) for p in self.params])

class ObservedData(DeferredActionCollection):
  def __init__(self):
    super(ObservedData,self).__init__(DEFAULT_PEST_DELIMITER,PTF_DATA_IO_INSTRUCTION_PREFIX)

  def all_files(self):
    return [instruction.pargs[0] for instruction in self.instructions]

  def copy_to(self,slave_dir):
    for fn in self.all_files():
      if os.path.basename(fn)==fn: # file is in cwd
        copyfile(fn,os.path.join(slave_dir,fn))

class CalibrationObservations(ConfigItemCollection):
  def __init__(self,veneer_prefix,delimiter,detailed_log=True):
    super(CalibrationObservations,self).__init__(OBS_DEFAULTS)
    self.detailed_log = detailed_log
    self.delimiter = delimiter
    self.data = ObservedData()
    self.instructions = []
    self.veneer_prefix = veneer_prefix

  def script(self):
    def store_ts(instruction):
      return "observed_ts.update(pd.%s.dropna(how='all').to_dict('series'))"%str(instruction)

    instructions = [
      'if failed:'
    ]

    instructions += [f'  pest_observations.append(("{obs["OBSNME"]}",{FAILED_RUN_OBJ_FN}))' for obs in self.items]
    instructions += [
      'else:'
    ]
    instructions += [f'  {i}' for i in self.instructions]

    return self.data.script(store_ts) +'\n' + '\n'.join(instructions)

  def compare(self,ts_name,mod_ref,stat=stats.nse,target=None,aggregation=None,time_period=None,obsnme=None,
              mod_scale=1,mod_transform='',mod_combine='sum',weight=1.0):
    if obsnme is None:
      obsnme = ts_name.replace(' ','_')

    if target is None:
      if hasattr(stat,'perfect'):
        target = stat.perfect
      else:
        target = 0

      if hasattr(stat,'maximise') and stat.maximise:
        if hasattr(stat,'perfect'):
          target = 0
        else:
          target = -target

    if aggregation is None:
      aggregation = 'daily'

    RETRIEVAL_TEMPLATE='mod_ts%s = %sretrieve_multiple_time_series(run_data=run_results,criteria=%s,timestep="%s")%s'
    if len(mod_transform) and not mod_transform.startswith('.'):
      mod_transform = '.' + mod_transform

    if isinstance(mod_ref,list):
      for i, ref in enumerate(mod_ref):
        ri = RETRIEVAL_TEMPLATE%(str(i),self.veneer_prefix,ref,aggregation,mod_transform)
        self.instructions.append(ri)
        self.instructions.append('mod_ts%d[mod_ts%d==-9999]=np.nan'%(i,i))
      combo_instruction = 'mod_ts = pd.DataFrame({"mod":%s})'%('+'.join(['mod_ts%d[mod_ts%d.columns[0]]'%(i,i) for i in range(len(mod_ref))]))
      self.instructions.append(combo_instruction)
    else:
      retrieval_instruction = RETRIEVAL_TEMPLATE%('',self.veneer_prefix,mod_ref,aggregation,mod_transform)
      self.instructions.append(retrieval_instruction)
      self.instructions.append('mod_ts[mod_ts==-9999]=np.nan')
    self.instructions.append('assert(not np.any(np.isnan(mod_ts)))')
    #self.instructions.append('print(mod_ts.columns)')
    #self.instructions.append('print(len(mod_ts.columns==0))')
    self.instructions.append('assert(len(mod_ts.columns==0))')
    self.instructions.append('mod_ts = mod_ts[mod_ts.columns[0]].dropna()%s'%('' if mod_scale==1 else ('*%f'%mod_scale)))
    self.instructions.append('obs_ts = observed_ts["%s"].dropna()'%ts_name)
    #self.instructions.append('print(len(obs_ts),obs_ts.index.dtype,mod_ts.index.dtype)')
    if time_period is not None:
      self.instructions.append('# Subset modelled and predicted')
      self.instructions.append('date_format = "%%Y/%%m/%%d"')
      self.instructions.append('t_start = pd.datetime.strptime("%s",date_format)'%time_period[0])
      self.instructions.append('t_end   = pd.datetime.strptime("%s",date_format)'%time_period[1])
      self.instructions.append('t_mask = (mod_ts.index>=t_start)&(mod_ts.index<=t_end)')
      self.instructions.append('mod_ts = mod_ts[t_mask]')

      self.instructions.append('t_mask = (obs_ts.index>=t_start)&(obs_ts.index<=t_end)')
      self.instructions.append('obs_ts = obs_ts[t_mask]')

#    self.instructions.append('print(obs_ts);print(mod_ts)')
    if (weight != 0.0) and hasattr(stat,'maximise') and stat.maximise:
      if hasattr(stat,'maximise'):
        sign = '%f-'%stat.maximise
      else:
        sign = '-'
    else:
      sign = '' 
    self.instructions.append('pest_observations.append(("%s",%s%s(obs_ts,mod_ts)))'%(obsnme,sign,stat.__name__))

    self.add(obsnme,target,weight)

  def penalise(self,penalty_name,penalty_equation,penalty_factor=1,multiplier=False):
    '''
    Add an objective that acts as a penalty on the solution based on inconsistency of
    parameter values. For example if DWC > EMC:

    case.observations.penalise('dwc_gt_emc','$DWC$ > $EMC$',1000)

    where:
     - penalty_name: name to use in PEST files to report the penalty value
     - penalty_equation: A string that, when processed by PEST will be valid Python. Can use
                             PEST parameter names, such as $EMC$.
                             Can be a boolean equation (if multiplier=False, default), in which case
                             the penalty will be penalty_factor if the equation evaluates True, 0 otherwise
                             Or, can be a numeric equation (if multiplier=True) in which case
                             the penalty will be penalty_factor * the value of the equation
         - penatly_factor: Value to use for the penalty (multiplier=False), or the value to multiply the
                           equation by (multiplier=True)
         - multiplier: if true, treat the equation as numeric and the penalty_factor as a multiplier, otherwise
                       treat the equation as boolean and the penalty_factor as the value to use if true
    '''
    if multiplier:
      instr = '%f * (%s)'%(penalty_factor,penalty_equation)
    else:
      instr = '%f if (%s) else 0'%(penalty_factor,penalty_equation)

    self.instructions.append('pest_observations.append(("%s",%s))'%(penalty_name,instr))
    self.add(penalty_name,0)

  def pif_line(self,obs):
    return '%s%s%s !%s!'%(self.delimiter,obs['OBSNME'],self.delimiter,obs['OBSNME'])

  def pif_text(self):
    return '\n'.join([PIF_PREFIX] + [self.pif_line(i) for i in self.items])

class Case(object):
  def __init__(self,name,optimiser='pest',model_servers=[9876],random_seed=1111,detailed_log=True):
    self.custom_imports = []
    self.optimiser=optimiser.lower()
    self.random_seed=random_seed
    self.name=name
    self.prefix = PTF_SINGLE_THREADED_INSTRUCTION_PREFIX
    self.pest_delimiter = DEFAULT_PEST_DELIMITER
    self.detailed_log = detailed_log
    self.parameters = CalibrationParameters(self.pest_delimiter,self.prefix,detailed_log)
    
    self.param_groups = ConfigItemCollection(PARA_GROUP_DEFAULTS)
    self.param_groups.add(DEFAULT_PG)

    self.observation_groups = ConfigItemCollection(OBS_GROUP_DEFAULTS)
    self.observation_groups.add(DEFAULT_OG)

    self.observations = CalibrationObservations(self.prefix,self.pest_delimiter,detailed_log)
    self.template_files = ConfigItemCollection(TEMPLATE_DEFAULTS)
    self.template_files.add(self.ptf_fn(),self.runner_fn())
    self.instruction_files = ConfigItemCollection(INSTRUCTION_DEFAULTS)
    self.instruction_files.add(self.pif_fn(),self.outputs_fn())

    self.options = CONTROL_DEFAULTS.copy()
    self.options.update(SVD_DEFAULTS)
    self.options['SIM_OPTIONS'] = {}
    self.expect_run_time = 1.0

    self.veneer_ports=model_servers

  def pif_fn(self):
    return "_%s_output.ins"%self.name

  def runner_fn(self):
    return "_run_%s.py"%self.name

  def ptf_fn(self):
    return "_run_%s.tpl"%self.name

  def outputs_fn(self):
    return '_%s_output.txt'%self.name

  def pcf_fn(self):
    return "%s.pst"%self.name

  def prf_fn(self):
    return "%s.rmf"%self.name

  def rec_fn(self):
    return "%s.rec"%self.name

  def par_fn(self):
    return "%s.par"%self.name

  def pcf_text(self):
    options = self.options.copy()
    options['PARAMETER_LINES'] = self.parameters.declarations()
    options['PARAMETER_GROUP_LINES'] = self.param_groups.declarations()
    options['TIED_PARAMETER_LINES'] = ''
    options['OBSERVATION_GROUP_LINES'] =self.observation_groups.declarations()
    options['OBSERVATION_LINES'] = self.observations.declarations()
    options['TEMPLATE_FILE_LINES'] = self.template_files.declarations()
    options['INSTRUCTION_FILE_LINES'] = self.instruction_files.declarations()

    options['NPAR'] = len(self.parameters)
    options['NOBS'] = len(self.observations)
    options['NTPLFLE'] = len(self.template_files)
    options['NINSFLE'] = len(self.instruction_files)
    options['RUNNER_FN'] = self.runner_fn()
    if options['MAXSING'] is None:
      options['MAXSING'] = options['NPAR']
    validate_dict(options)
    return PCF.substitute(options) 

  def ptf_imports(self):
    if not len(self.custom_imports):
      return ''
    result = [
      '',
      '# Custom imports',
      'import sys'
    ]
    directories = set([os.path.dirname(i.__file__) for i in self.custom_imports])
    result += [f'sys.path.append(r"{d}")' for d in directories]
    result += [f'from {i.__name__} import *' for i in self.custom_imports]
    result += ['']
    return '\n'.join(result)

  def ptf_text(self):
    full = (PTF_PREFIX%self.detailed_log + self.ptf_imports() + self.parameters.script() + PTF_RUN + self.observations.script() + PTF_ASSESS)

    if self.detailed_log:
      full += '\n' + self.parameters.log_script() + WRITE_DETAILED_LOG

    return full%(self.options['SIM_OPTIONS'],self.outputs_fn())


  def pif_text(self):
    return self.observations.pif_text()

  def prf_text(self):
    prf_options = PRF_DEFAULTS.copy()
    prf_options['NSLAVE'] = len(self.veneer_ports)
    prf_options['SLAVE_LINES'] = '\n'.join(['%s %s\\'%(slave,os.path.join('.',slave)) for slave in [self.slave_name(p) for p in self.veneer_ports]])
    prf_options['SLAVE_RUNTIME_LINES'] = ' '.join(['%f'%self.expect_run_time]*len(self.veneer_ports))
    return PRF.substitute(prf_options)

  def slave_name(self,p):
    return 'Slave_%d'%p

  def stdio_params(self,**kwargs):
    p={}
    p['min_relative_objective_fn_change'] = 0.001
    p['iterations_for_change']=5
    p['max_iters']=50000
    random_seed = self.random_seed
    if random_seed is None:
      random_seed = np.random.randint(2**15)
    random_seed = str(random_seed)
    if self.optimiser == 'sceua_p':
      p.update(kwargs)
      sce_params = [max(10,len(self.veneer_ports)),5,9,5,'y',9,random_seed,'n',p['min_relative_objective_fn_change'],p['iterations_for_change'],p['max_iters']]
      if len(self.observations)==1:
        sce_params = ['i']+sce_params
      return '\n'.join([str(p) for p in sce_params])
    elif self.optimiser == 'cmaes_p':
      p['iterations_for_change']=40
      p['min_relative_param_change']=0.001
      p['rel_hi_low_object_fn']=0.01
      p['no_iterations_hi_lo']=10
      lambda_param = int(4 + 3*math.log(len(self.parameters)))
      p['lambda_param'] = max(lambda_param,len(self.veneer_ports))
      p['mu_param'] = int(p['lambda_param']/2)
      p.update(kwargs)
      cmaes_params = [p['lambda_param'],p['mu_param'],'s',random_seed,'n']
      
      if len(self.observations) > len(self.parameters):
        cmaes_params += ['n'] # Employ SVD hybridisation?

      cmaes_params += ['n',p['min_relative_objective_fn_change'],
                      p['iterations_for_change'],p['min_relative_param_change'],p['iterations_for_change'],
                      p['rel_hi_low_object_fn'],p['no_iterations_hi_lo'],p['max_iters'],'y']

      # lambda, mu, recombination weights, random seed, read covariance matrix from file, forgive model runs
      # ... named...
      # run model with initial,
      if len(self.observations)==1:
        cmaes_params = ['i'] + cmaes_params
      return '\n'.join([str(p) for p in cmaes_params]) + '\n'
    return None

  def write_connection_file(self,wd,port):
    with open(os.path.join(wd,CONNECTION_FN),'w') as f:
      f.write(str(port))

  def get_clients(self):
    return BulkVeneer(self.veneer_ports)

  def run(self,**kwargs):
    stdio = self.stdio_params(**kwargs)
    if stdio is None:
      redirect = ''
    else:
      redirect = ' < pest_stdio.txt'
      open('pest_stdio.txt','w').write(stdio)

    open(self.pif_fn(),'w').write(self.pif_text())
    open(self.ptf_fn(),'w').write(self.ptf_text())
    open(self.pcf_fn(),'w').write(self.pcf_text())

    opt_exe=self.optimiser
    opt_args=''

    if len(self.veneer_ports)==1:
      working_dir='.'
      self.write_connection_file(working_dir,self.veneer_ports[0])
      if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    else:
      if opt_exe=='pest':
        opt_exe = 'ppest'
      else:
        opt_args=' /p'

      # +++ Only for parallel pest
      open(self.prf_fn(),'w').write(self.prf_text())

      cwd = os.getcwd()
      slave_processes = []
      for slave in self.veneer_ports:
        os.chdir(cwd)
        name = self.slave_name(slave)
        slave_dir = os.path.join('.',name)
        if not os.path.exists(slave_dir):
          os.mkdir(slave_dir)
        else:
          slave_log = os.path.join(slave_dir,LOG_FILE)
          if os.path.exists(slave_log):
            os.remove(slave_log)
        self.observations.data.copy_to(slave_dir)

        if self.optimiser == 'sceua_p':
          for fn in [self.pcf_fn(),self.pif_fn(),self.ptf_fn()]:
            copyfile(fn,os.path.join(slave_dir,fn))
        self.write_connection_file(slave_dir,slave)

        slave_process = Popen(['pslave'],stdin=PIPE,cwd=slave_dir)
        slave_instruction = 'python %s\n'%self.runner_fn()
        if self.optimiser == 'sceua_p':
          slave_instruction = 'sceua_p %s /s\n'%self.name
        slave_process.stdin.write(bytes(slave_instruction,'utf-8'))
        #slave_process.stdin.write(bytes('dir\n','utf-8'))
#        slave_process.stdin.write(bytes('set /p x=y\n','utf-8'))
        slave_process.stdin.flush()
        slave_processes.append(slave_process)

      kill_all_on_exit(slave_processes)

    os.system('%s %s %s %s'%(opt_exe,self.pcf_fn(),opt_args,redirect))
    kill_all_now(slave_processes)

    return self.get_results()

  def get_results(self):
    result = {}
    fn = self.rec_fn()
    txt = open(fn).read()

    params_txt = open(self.par_fn()).read().splitlines()
    columns = ['parameter','value','scale','offset']
    param_vals = [dict(zip(columns,line.strip().split())) for line in params_txt[1:]]
    params = pd.DataFrame(param_vals)
    params = params.set_index('parameter')
    for col in columns[1:]:
      params[col] = params[col].astype('f')

    result['results_file']=fn
    result['text']=txt
    result['parameters'] = params

    if self.detailed_log:
      result['log'] = self.read_logs()
    return result

  def apply_parameters(self,v=None,parameters=None):
    '''
    Apply a calibrated parameter set to a Source model at the end of a Veneer end-point.

    Used post-calibration.

    Parameters:

    * v - Veneer client object. If None, parameters will be applied to ALL veneer instanced used 
          in the calibration
    * parameters - dictionary-like object of parameters. If None, used the calibrated parameters
                   from the simulation.
    '''
    if parameters is None:
      results = self.get_results()
      parameters = results['parameters'].value # transform to dict-like

    if not isinstance(parameters,dict):
      parameters = parameters.to_dict()

    if v is None:
      v = self.get_clients()
    
    self.parameters.eval_script(parameters,{'v':v})

    
  def read_logs(self):
    if len(self.veneer_ports)==1:
      return pd.read_csv(LOG_FILE)
    else:
      all_logs = [pd.read_csv(os.path.join('Slave_%d'%p,LOG_FILE)) for p in self.veneer_ports]
      return pd.concat(all_logs,axis=0,ignore_index=True)

def make_pest_name(name):
  if len(name)<=12:
    return name

  orig_name = name
  subset=5
  while len(name)>12 and subset>0:
    pieces = orig_name.split('_')
    name = ''.join(p[:subset] for p in pieces)
    subset -= 1

  return name[:12]

# TODO
# Shutdown Veneer with POST /shutdown
# Ability to queue up IronPython requests... to reduce latency...
# start source n times with a given project



#### Disable recording... Enable relevant recording...
#### Why no modelled results?


'''
Parallel PEST / BeoPEST

BeoPEST (TCPIP) only available for main pest. not for CMAES_P. not for SCE?

Need to identify file dependencies and copy them to a working  directory for each instance (unless the filenames are absolute paths)

Need to start slave in each working directory...
  Pipe in name of model executable.
  How to communicate the Veneer end point? Write to file in that directory?
  /n - no shutdown

Don't need PEST distributed across the network. Can run as many instances of PSLAVE locally communicating to different Source end points

construct Case instance with list of instances? Do we need to shut them down at the end?


option to stop pest... and to run asynchronously...

'''