"""
Module that hosts several important classes
"""

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import sys, traceback

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DBGateway(object):
	"""
	The gateway class of Database. As we go more into the abstraction of database,
	this class handles the communications with the DB. 

	The gateway class hosts mostly statically defined methods and properties
	"""

	# Database credentials
	DB_HOST = "localhost"
	DB_NAME = "open_database"
	DB_USER = "root"
	DB_PASS = ""

	# Utilized table(s)
	TBL_HOUSE_INFO = 'ps_house_info'
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

	# Statically defined database engine
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

	@staticmethod
	def recordAgent(agent_addr, agent_type):
		"""
		Store the agent in DB.
		Assume that there will be only one active
		blockchain agent.

		The recording of agents into database should potentially go to a different
		agent. We will ove the functionalities later to a "Service" agent.
		"""

		# For now, just use the raw query
		query = "INSERT INTO `{}` (`session_id`, `agent_id`, `agent_address`, "\
				"`agent_type`, `status`, `insertion_datetime`) VALUES ('rootcontainersession', '-1', "\
				"'{}', '{}', 'active', CURRENT_TIMESTAMP)".format(Configure.TBL_AGENTS, agent_addr, agent_type)

		# Execute the query
		try:
			result = Configure.get_db_engine().execute(query)
			logger.info("The agent is recorded successfully")
		except Exception as e:
			logger.info("Query failed to execute!")
			traceback.print_exc(file=sys.stdout)
			return False

		return True	

	def killAgent(agent_addr, agent_type):
		"""
		Kill any agent by changing the status to 'dead'.
		This will also probably move to the Service agent we talked about 
		at last function definition.
		"""

		query = "UPDATE `{}` set `status`='dead' WHERE agent_address = '{}' "\
				"AND agent_type='{}' AND `status`='active'".format(Configure.TBL_AGENTS, agent_addr, agent_type)

		try:
			Configure.get_db_engine().execute(query)
		except Exception as e:
			logger.info("Query failed to execute!")
			traceback.print_exc(file=sys.stdout)
			return False

		return True	




