"""
This module contains the Home Container that hosts
home agents.

"""
import aiomas
from agents.BlockchainAgent import BlockchainAgent
from agents.HomeAgent import HomeAgent
from sqlalchemy import create_engine
import sys, traceback

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def runContainer():	
	# Creating container contains the home agents
	# and prediction agents
	HC = aiomas.Container.create(('localhost', 5556))

	# List of Homes
	homes = [9019,  7881, ]#  100237, 7850, 980, 9981,]

	# Create the DB engine
	db_engine = create_engine("mysql+pymysql://root@127.0.0.1/open_database")

	# Address of the blockchain agent
	# Later, it will be retreived from the Agent Server
	bc_address = "tcp://localhost:5555/0"

	# Initiate the agents into HC
	homeAgents = [HomeAgent(container=HC, agent_id=home, db_engine=db_engine, bc_address=bc_address) for home in homes]

	for homeAgent in homeAgents:
		logger.info("{}. {}".format(homeAgent.agent_id, homeAgent.addr))

	# Run the event loop
	try:
		logger.info("Running the event loop. One of the home agents trying to connect with BC agent!")
		# aiomas.run(until=homeAgents[0].communicateBlockchain(bc_address))
		aiomas.run()

	except Exception as e:
		traceback.print_exc(file=sys.stdout)

	# Shutting donw the controller and thereby cleaning 
	# all agents
	try:
		logger.info("Shutting down the home container...")
		HC.shutdown()
		logger.info("Done.")
	except Exception as e:
		logger.info("Failed to shutdonw the home container")
		traceback.print_exc(file=sys.stdout)


if __name__ == '__main__':
	runContainer()
