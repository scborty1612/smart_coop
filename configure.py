"""
Configuration
"""

from sqlalchemy import create_engine

class Configure(object):
	"""
	Some globally accessable constants
	"""
	GRANULARITY = 15 # mins

	# Simulation start datetime
	SIM_START_DATETIME = "2015-01-15 00:00:00"
	SIM_END_DATETIME = "2015-01-16 00:00:00"

	# Delay to impose during each interval
	DELAY = 0.10

	# Battery characteristics
	BATTERY_CHAR = {'capacity' : 8.,  # in kWh 
               	    'd_rating' :  3., # in kW
               	    'c_rating' :  1.5,  # in kW
               	    'soc_high' : 0.80,
               	    'soc_low' : 0.20,
               	    'c_eff' : 0.92,
               	    'd_eff' : 0.94}