"""
Configuration
"""

from sqlalchemy import create_engine

class Configure(object):
	"""
	Some globally accessable constants
	"""
	DB_HOST = "localhost"
	DB_NAME = "open_database"
	DB_USER = "root"
	DB_PASS = ""

	db_engine = None

	@staticmethod
	def get_db_engine():
		if Configure.db_engine is not None:
			return Configure.db_engine
		Configure.db_engine = create_engine("mysql+pymysql://{}@{}/{}".format(Configure.DB_USER, 
			Configure.DB_HOST, Configure.DB_NAME))
		return Configure.db_engine

