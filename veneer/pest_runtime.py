#from .general import Veneer
#from .pest import *

CONNECTION_FN='veneer_connection.txt'

def find_port():
	with open(CONNECTION_FN,'r') as f:
		return int(f.read())

def write_outputs(outputs,fn):
	with open(fn,'w') as dest:
		for k,v in outputs:
			dest.write('%s %f\n'%(k,v))