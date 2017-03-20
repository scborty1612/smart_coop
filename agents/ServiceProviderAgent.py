"""
The "ServiceProviderAgent" provides a kind of "yellow-page"
service to the agent environment. 

Basically, the current functionalities of this agent is to store agents'
information into a database and retreive it whenever its necessary.

"""

# Import the agent package
import aiomas

# For storing and manipulating data
import pandas as pd
import numpy as np
import uuid


import matplotlib.pyplot as plt
import seaborn as sns

# Adding the configuration script
import sys
sys.path.append("../")
from configure import Configure as CF
from util import DBGateway as DB

# Logging stuffs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys, traceback

class ServiceProviderAgent(aiomas.Agent):
	"""
	The residential agents
	"""
	def __init__(self, container, ):
		super().__init__(container)
		# print(self.container._base_url)

		# Registering itself
		session_id = uuid.uuid4().hex

		status = self.recordAgent(session_id=str(session_id),
					  container_name='rootcontainer',
					  container_address=self.container._base_url,
					  agent_id=-1,
					  agent_address=self.addr,
					  agent_type='service_provider_agent',
					  agent_functionality='Service Provider')

		if status:
			self.session_id = session_id
		else:
			self.session_id = 'invalid_session'


	@aiomas.expose
	async def killAgent(self, agent_id):
		"""
		Deactivate an agent by agent type
		"""
		query = "UPDATE `{}` set `agent_status`='dead' where session_id='{}' AND "\
				"agent_id={}".format(DB.TBL_AGENTS_SERVICE, self.session_id, agent_id)

		logging.info("Killing agent {}".format(agent_id))
		try:
			DB.get_db_engine().execute(query)
		except Exception as e:
			logger.info("Query failed to execute!")
			# traceback.print_exc(file=sys.stdout)
			return False

		return True

	@aiomas.expose
	def getAliveSessionID(self):
		"""
		Provide alive and recent session id
		"""
		query = "SELECT `session_id` from {} WHERE `agent_status`='alive' ORDER BY `insertion_datetime` DESC".format(DB.TBL_AGENTS_SERVICE)

		# Dump the reseult into a dataframe
		df = pd.read_sql(query, DB.get_db_engine())

		if len(df) < 1:
			return False

		return df['session_id'][0]	

	@aiomas.expose
	def getAliveBlockchain(self, session_id):
		"""
		Provide alive and recent session id
		"""
		query = "SELECT `agent_address` from {} WHERE `session_id`='{}' and `agent_status`='alive' and "\
				"`agent_type`='Blockchain' ORDER BY `insertion_datetime` DESC".format(DB.TBL_AGENTS_SERVICE, session_id)

		# Dump the reseult into a dataframe
		df = pd.read_sql(query, DB.get_db_engine())

		if len(df) < 1:
			return False

		return df['agent_address'][0]	



	@aiomas.expose
	def recordAgent(self, **kwargs):

		columns = ",".join(["`{}`".format(c) for c in kwargs.keys()])
		values = ",".join(["'{}'".format(v) for v in kwargs.values()])


		# For now, just use the raw query
		query = "INSERT INTO `{}` ({}) VALUES ({})".format(DB.TBL_AGENTS_SERVICE, columns, values)
		# logger.info(query)

		# Execute the query
		try:
			result = DB.get_db_engine().execute(query)
			logger.info("The agent is recorded successfully")
		except Exception as e:
			logger.info("Query failed to execute!")
			traceback.print_exc(file=sys.stdout)
			return False

		return True	

