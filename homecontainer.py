"""
This module contains the Home Container that hosts
home agents.

"""
import aiomas
from agents.BlockchainAgent import BlockchainAgent
from agents.HomeAgent import HomeAgent
from configure import Configure as CF

# Database related stuff
from sqlalchemy import create_engine

import pandas as pd

# System and traceback
import sys, traceback

# Logging stuffs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clock related stuff
import asyncio
import time
import datetime
import random
import uuid

# Agree on clocking
CLOCK = aiomas.ExternalClock(CF.SIM_START_DATETIME)

# Setting up the clock setting function
async def clock_setter(factor=10.):
	while True:
		await asyncio.sleep(factor)

		# delta t is 15min
		CLOCK.set_time(CLOCK.time() + (1*60*15))

def createSession(agents, db_engine=None):
	"""
	Create a session with the home agents.
	"""
	# Unique UUID as session ID
	session_id = uuid.uuid4().hex

	# Prepare the dataframe to dump into a mysql table
	agent_ids = []
	agent_adds = []
	for agent in agents:
		agent_ids.append(agent.agent_id)
		agent_adds.append(agent.addr)

	df = pd.DataFrame(data=[agent_ids, agent_adds],)
	df = df.T
	df.columns = ['agent_id', 'agent_address']
	df['session_id'] = session_id

	df.to_sql(name='{}'.format(CF.TBL_AGENTS), con=db_engine, if_exists='append', index=False, chunksize=200)

	return session_id

def killSession(session_id, db_engine=None):
	"""
	Kill the session from DB.
	"""
	query = "UPDATE `{}` set `status`='dead' where session_id='{}'".format(CF.TBL_AGENTS, session_id)
	try:
		db_engine.execute(query)
	except Exception as e:
		logger.info("Query failed to execute!")
		traceback.print_exc(file=sys.stdout)
		return False

	return True

def getActiveBlockchainAddress(db_engine):
	"""
	Retreive the active blockchain address
	from DB.
	"""
	query = "SELECT `agent_address` FROM `{}` WHERE `agent_type`='blockchain' AND "\
			" `status`='Active'".format(CF.TBL_AGENTS)

	# Dump the reseult into a dataframe
	df = pd.read_sql(query, db_engine)

	if len(df) < 1:
		return None

	return df['agent_address'][0]	

def runContainer():	

	# Creating container contains the home agents
	# and prediction agents
	HC = aiomas.Container.create(('localhost', 5556), clock=CLOCK)
	# HC = aiomas.Container.create(('localhost', 5556), clock=CLOCK)

	# Set the clcok
	# t_clock_setter = asyncio.async(clock_setter())


	# List of Homes
	homes = [9019, 7850, ]# 7881, 100237, 9981, 980,]

	# Create the DB engine
	# db_engine = create_engine("mysql+pymysql://{}@{}/{}".format(CF.DB_USER, CF.DB_HOST, CF.DB_NAME))
	db_engine = CF.get_db_engine()

	# Initiate the agents into HC
	homeAgents = [HomeAgent(container=HC, agent_id=home, db_engine=db_engine,) for home in homes]

	# Creating the session
	session_id = createSession(agents=homeAgents, db_engine=db_engine)

	# Address of the blockchain agent
	# Later, it will be retreived from the Agent Server
	bc_address = getActiveBlockchainAddress(db_engine=db_engine)

	if bc_address is None:
		logging.info("Blockchain is not initiated.")
	else:
		# Bind the blockchain with home agents
		for agent in homeAgents:
			agent.setBlockchainAddress(bc_address=bc_address)

	# Run the event loop
	try:
		logger.info("Running the event loop. One of the home agents is trying to connect with BC agent!")
		logger.info("Session ID:{}".format(session_id))

		# Run the even loop 
		aiomas.run()
	except KeyboardInterrupt:
		logger.info("Stopping the event loop")
		# Try to stop the event loop

	except Exception as e:
		traceback.print_exc(file=sys.stdout)
	finally:
		# Killing the current session
		killSession(session_id=session_id, db_engine=db_engine)
		

	# Shutting donw the controller and thereby cleaning 
	# all agents
	try:
		logger.info("Shutting down the home container...and cancelling the clock")
		HC.shutdown()
		# t_clock_setter.cancel()
		logger.info("Done.")
	except Exception as e:
		logger.info("Failed to shutdown the home container")
		traceback.print_exc(file=sys.stdout)


if __name__ == '__main__':
	runContainer()
