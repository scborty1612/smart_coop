"""
Module that hosts several important classes.
Currently, hosting
1. DBGateway class: The class provides interface to the current database
utilized in the system.

"""

import pandas as pd
import numpy as np
import uuid

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
	DB_NAME = "pecan_street"
	DB_USER = "root"
	DB_PASS = ""

	# Utilized table(s)
	TBL_HOUSE_INFO = 'ps_house_info'
	TBL_ENERGY_USAGE = 'ps_energy_usage_15min'
	TBL_AGENTS = 'tbl_agents'
	TBL_AGENTS_SERVICE = 'tbl_agent_service'

	# Columns for TBL agents (home agent)
	TBL_AGENTS_INDEX = 'timestamp'
	TBL_AGENTS_AGENT_ID_COL = 'house_id'

	# The columns for generation (PV) and demand, respectively.
	# Make sure to keep the ordering
	TBL_AGENTS_COLS = ['gen', 'use']

	"""
	Constants related to data scaling (PV) and demand, respectively
	"""
	TBL_AGENTS_COLS_SCALE = ['1.0', '1.0']

	# Statically defined database engine
	db_engine = None

	# Create a single database connnector
	# and re-use
	@staticmethod
	def get_db_engine():
		if DBGateway.db_engine is not None:
			return DBGateway.db_engine
		try:
			DBGateway.db_engine = create_engine("mysql+pymysql://{}@{}/{}".format(DBGateway.DB_USER, 
				DBGateway.DB_HOST, DBGateway.DB_NAME))
		except Exception as e:
			raise Exception("Can't connect to DB.")

		return DBGateway.db_engine


	@staticmethod
	def getServiceProviderAgent(session_id):
		"""
		Retreive the active Service Provider agent
		from DB.
		"""
		query = "SELECT `agent_address` FROM `{}` WHERE `agent_type`='service_provider_agent' AND "\
				" `agent_status`='alive' order by `insertion_datetime`".format(DBGateway.TBL_AGENTS_SERVICE)

		# Dump the reseult into a dataframe
		df = pd.read_sql(query, DBGateway.get_db_engine())

		if len(df) < 1:
			return None

		return df['agent_address'][0]	

	@staticmethod
	def killAgent(agent_addr, agent_type):
		"""
		Kill any agent by changing the status to 'dead'.
		This will also probably move to the Service agent we talked about 
		at last function definition.
		"""

		query = "UPDATE `{}` set `status`='dead' WHERE agent_address = '{}' "\
				"AND agent_type='{}' AND `status`='active'".format(DBGateway.TBL_AGENTS, agent_addr, agent_type)

		try:
			DBGateway.get_db_engine().execute(query)
		except Exception as e:
			logger.info("Query failed to execute! Work with CSV data!")
			traceback.print_exc(file=sys.stdout)
			return True

		return True	

	@staticmethod
	def getAgentInfo(agent_id=None, session_id=None):
		"""
		Retrieve the home agent's address given the
		session id and agent id
		"""
		# Statement to retreive agent address
		query = "SELECT `agent_address`, `agent_type` FROM `{}` WHERE `agent_id`={} AND "\
				"session_id='{}' AND `agent_status`='alive'".format(DBGateway.TBL_AGENTS_SERVICE, agent_id, session_id)
		
		# print(query)
		# Dump the reseult into a dataframe
		df = pd.read_sql(query, DBGateway.get_db_engine())

		if len(df) < 1:
			return None

		return df['agent_address'][0], df['agent_type'][0]


	@staticmethod
	def loadGenAndDemandDataForHome(agent_id, start_datetime="2015-01-01 00:00:00", end_datetime="2015-01-31 00:00:00"):
		"""
		Load the household's generation and demand data for a particular period
		"""

		# Form the where clause based on the date filtering
		whereClause = "{} = {}".format(DBGateway.TBL_AGENTS_AGENT_ID_COL, agent_id) 

		if start_datetime and end_datetime:
			whereClause += " AND date_format(`{}`, '%%Y-%%m-%%d %%H:%%i:%%s') >= '{}' "\
						   " AND date_format(`{}`, '%%Y-%%m-%%d %%H:%%i:%%s') < '{}' ".format(DBGateway.TBL_AGENTS_INDEX, start_datetime, 
						   														  DBGateway.TBL_AGENTS_INDEX, end_datetime)
		
		# Form the sql query to fetch residential data
		scaled_cols = ",".join(["`{}`*{} as `{}`".format(col, scale, col) 
								for (col, scale) in zip(DBGateway.TBL_AGENTS_COLS, DBGateway.TBL_AGENTS_COLS_SCALE)])
		
		sql_query = "SELECT `{}`, {} FROM `{}` where {}".format(DBGateway.TBL_AGENTS_INDEX, scaled_cols, DBGateway.TBL_ENERGY_USAGE, whereClause)

		# Fetch the data into a pandas dataframe
		df = pd.read_sql(sql_query, DBGateway.get_db_engine(), parse_dates=[DBGateway.TBL_AGENTS_INDEX], index_col=[DBGateway.TBL_AGENTS_INDEX])

		# df['int_timestamp'] = df['timestamp'].apply(lambda x:int(x.timestamp()))

		if len(df) <= 2:
			# Apparently, no data is there
			return None

		# The columns containing devices
		# consumption_cols = ['air1', 'air2', 'air3', 'airwindowunit1', 'aquarium1', 'bathroom1', 'bathroom2', 
		# 		'bedroom1', 'bedroom2', 'bedroom3', 'bedroom4', 'bedroom5', 'car1', 'clotheswasher1', 
		# 		'clotheswasher_dryg1', 'diningroom1', 'diningroom2', 'dishwasher1', 'disposal1', 'drye1', 
		# 		'dryg1', 'freezer1', 'furnace1', 'furnace2', 'garage1', 'garage2', 'heater1', 
		# 		'housefan1', 'icemaker1', 'jacuzzi1', 'kitchen1', 'kitchen2', 'kitchenapp1', 'kitchenapp2', 
		# 		'lights_plugs1', 'lights_plugs2', 'lights_plugs3', 'lights_plugs4', 'lights_plugs5', 'lights_plugs6', 
		# 		'livingroom1', 'livingroom2', 'microwave1', 'office1', 'outsidelights_plugs1', 'outsidelights_plugs2', 
		# 		'oven1', 'oven2', 'pool1', 'pool2', 'poollight1', 'poolpump1', 'pump1', 'range1', 'refrigerator1', 
		# 		'refrigerator2', 'security1', 'shed1', 'sprinkler1', 'utilityroom1', 'venthood1', 'waterheater1', 
		# 		'waterheater2', 'winecooler1']

		# For now, reduce the DF size 
		# by removing individual loads
		# for col in consumption_cols:
		# 	del df[col]

		# The PV columns may contain negative values 
		# For now, update those values with zero
		gen_col = DBGateway.TBL_AGENTS_COLS[0]
		df[gen_col] = df[gen_col].apply(lambda x: max(x, 0))

		# Load after PV integration (if there is any)
		df['load'] = df[DBGateway.TBL_AGENTS_COLS[1]]-df[DBGateway.TBL_AGENTS_COLS[0]]
		
		logging.info("{}. Total number of records: {}".format(agent_id, len(df)))

		return df

