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
CLOCK = aiomas.ExternalClock("2015-01-15T00:00:00")

# Setting up the clock setting function
async def clock_setter(factor=0.25):
	while True:
		await asyncio.sleep(factor)

		# delta t is 15min
		CLOCK.set_time(CLOCK.time() + (1*60*15))

def createSession(agents, db_engine=None):
	"""
	Create a session with the home agents.
	"""
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

	df.to_sql(name='tbl_agents', con=db_engine, if_exists='append', index=False, chunksize=200)

	return session_id

def runContainer():	

	# Creating container contains the home agents
	# and prediction agents
	HC = aiomas.Container.create(('localhost', 5556), clock=CLOCK)

	# Set the clcok
	t_clock_setter = asyncio.async(clock_setter())


	# List of Homes
	homes = [9019,  7881, 100237, 7850, 980, 9981,]

	# Create the DB engine
	# db_engine = create_engine("mysql+pymysql://{}@{}/{}".format(CF.DB_USER, CF.DB_HOST, CF.DB_NAME))
	db_engine = CF.get_db_engine()

	# Address of the blockchain agent
	# Later, it will be retreived from the Agent Server
	bc_address = "tcp://localhost:5555/0"

	# Initiate the agents into HC
	homeAgents = [HomeAgent(container=HC, agent_id=home, db_engine=db_engine, bc_address=bc_address) for home in homes]

	# Creating the session
	session_id = createSession(agents=homeAgents, db_engine=db_engine)

	# Run the event loop
	try:
		logger.info("Running the event loop. One of the home agents trying to connect with BC agent!")
		logger.info("Session ID:{}".format(session_id))
		# aiomas.run(until=homeAgents[0].communicateBlockchain(bc_address))
		aiomas.run()

	except Exception as e:
		traceback.print_exc(file=sys.stdout)

	# Shutting donw the controller and thereby cleaning 
	# all agents
	try:
		logger.info("Shutting down the home container...and cancelling the clock")
		HC.shutdown()
		t_clock_setter.cancel()
		logger.info("Done.")
	except Exception as e:
		logger.info("Failed to shutdonw the home container")
		traceback.print_exc(file=sys.stdout)


if __name__ == '__main__':
	runContainer()
