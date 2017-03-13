"""
Configuration
"""

from sqlalchemy import create_engine

class Configure(object):
	"""
	Some globally accessable constants
	"""

	# Database credentials
	DB_HOST = "localhost"
	DB_NAME = "open_database"
	DB_USER = "root"
	DB_PASS = ""

	# Utilized table(s)
	TBL_ENERGY_USAGE = 'ps_energy_usage_15min'
	TBL_AGENTS = 'tbl_agents'

	# Columns for TBL agents (home agent)
	TBL_AGENTS_INDEX = 'timestamp'
	TBL_AGENTS_AGENT_ID_COL = 'house_id'

	# The columns for generation (PV) and demand, respectively.
	# Make sure to keep the ordering
	TBL_AGENTS_COLS = ['gen', 'grid']

	"""
	Constants related to data scaling (PV) and demand, respectively
	"""
	TBL_AGENTS_COLS_SCALE = ['1.0', '1.0']

	granularity = 15 # mins

	# Simulation start datetime
	SIM_START_DATETIME = "2015-01-15 00:00:00"
	SIM_END_DATETIME = "2015-01-16 10:00:00"

	db_engine = None

	# Create only one DB connection
	# and re-use it (singleton)
	@staticmethod
	def get_db_engine():
		if Configure.db_engine is not None:
			return Configure.db_engine
		try:
			Configure.db_engine = create_engine("mysql+pymysql://{}@{}/{}".format(Configure.DB_USER, 
				Configure.DB_HOST, Configure.DB_NAME))
		except Exception as e:
			raise Exception("Can't connect to DB.")

		return Configure.db_engine



