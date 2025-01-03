from src.parser import args

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Data folders
output_folder = '.\\processed'

# Threshold parameters
lm_d = {
		'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
		'MBA': [(0.87, 1), (0.93, 1.04)],
        'SMAP': [(0.999, 1.04), (0.98, 1)],
		'MSL': [(0.97, 1), (0.999, 1.04)],
        'WADI': [(0.99, 1), (0.999, 1)],
        
		'SWaT': [(0.993, 1), (0.993, 1)],
		'UCR': [(0.993, 1), (0.99935, 1)],
		'NAB': [(0.991, 1), (0.99, 1)],
		
		'MSDS': [(0.91, 1), (0.9, 1.04)],
        'synthetic': [(0.999, 1), (0.999, 1)],
	}
lm = lm_d[args.dataset][0]
# lm = lm_d[args.dataset][1]
"""
MBA 1 1  
"""

# Hyperparameters
lr_d = {
		'SMD': 0.0001, 
		'synthetic': 0.0001, 
		'SWaT': 0.008, 
		'SMAP': 0.001, 
		'MSL': 0.002, 
		'WADI': 0.0001, 
		'MSDS': 0.001, 
		'UCR': 0.006, 
		'NAB': 0.009, 
		'MBA': 0.001, 
	}
lr = lr_d[args.dataset]

# Debugging
percentiles = {
		'SMD': (98, 2000),
		'synthetic': (95, 10),
		'SWaT': (95, 10),
		'SMAP': (97, 5000),
		'MSL': (97, 150),
		'WADI': (99, 1200),
		'MSDS': (96, 30),
		'UCR': (98, 2),
		'NAB': (98, 2),
		'MBA': (99, 2),
	}
percentile_merlin = percentiles[args.dataset][0]
cvp = percentiles[args.dataset][1]
debug = 9