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

	granularity = 15 # mins

	# Simulation start datetime
	SIM_START_DATETIME = "2015-01-15 00:00:00"
	SIM_END_DATETIME = "2015-01-16 00:00:00"

	db_engine = None

	# Create only one DB connection
	# and re-use it (singleton)
	@staticmethod
	def get_db_engine():
		if Configure.db_engine is not None:
			return Configure.db_engine
		Configure.db_engine = create_engine("mysql+pymysql://{}@{}/{}".format(Configure.DB_USER, 
			Configure.DB_HOST, Configure.DB_NAME))
		return Configure.db_engine

