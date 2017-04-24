"""
Configuration
"""

from sqlalchemy import create_engine

class Configure(object):
	# Mode of DB connection
	MODE = 'LOCAL'
	# MODE = 'REMOTE'

	"""
	Some globally accessable constants
	"""
	GRANULARITY = 15 # mins

	# Simulation start datetime
	SIM_START_DATETIME = "2015-01-15 00:00:00"
	SIM_END_DATETIME = "2015-02-17 00:00:00"

	# Delay to impose during each interval
	DELAY = 0.2

	# Battery characteristics
	BATTERY_CHAR = {'capacity': 7.,  # in kWh 
               	    'd_rating': 5., # in kW
               	    'c_rating': 3,  # in kW
               	    'soc_high': 0.88,
               	    'soc_low': 0.12,
               	    'c_eff': 0.95,
               	    'd_eff': 0.95}