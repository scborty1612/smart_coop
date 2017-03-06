
"""
This module contains the Root Container that hosts
"Blockchain" agent.

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
	RC = aiomas.Container.create(('localhost', 5555))

	# Create the DB engine
	db_engine = create_engine("mysql+pymysql://root@127.0.0.1/open_database")

	# Initiate the blockchain agent
	blockChainAgent = BlockchainAgent(container=RC, db_engine=db_engine)

	# Dump the blockchain agent address
	logger.info("Blcokchain agent initiated at {}".format(blockChainAgent.addr))

	# Run the event loop
	try:
		logger.info("Running the event loop. The blockchain agent is open to be connected!")
		aiomas.run()
	except Exception as e:
		traceback.print_exc(file=sys.stdout)
	# Shutting donw the controller and thereby cleaning 
	# all agents
	try:
		logger.info("Shutting down the root container...")
		RC.shutdown()
		logger.info("Done.")
	except Exception as e:
		logger.info("Failed to shutdonw the root container")
		traceback.print_exc(file=sys.stdout)


if __name__ == '__main__':
	runContainer()
