
"""
This module contains the Root Container that hosts
"Blockchain" agent.

"""
import aiomas
from agents.BlockchainAgent import BlockchainAgent
from agents.HomeAgent import HomeAgent
from configure import Configure as CF

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
	# db_engine = create_engine("mysql+pymysql://{}@{}/{}".format(CF.DB_USER, CF.DB_HOST, CF.DB_NAME))

	# Initiate the blockchain agent
	blockChainAgent = BlockchainAgent(container=RC, )

	# Dump the blockchain agent address
	logger.info("Blcokchain agent initiated at {}".format(blockChainAgent.addr))

	# Run the event loop
	try:
		logger.info("Running the event loop. The blockchain agent is open to be connected!")
		aiomas.run()
	except KeyboardInterrupt:
		logging.info("Keyboard Interrupted")		
	except Exception as e:
		traceback.print_exc(file=sys.stdout)
	# Shutting donw the controller and thereby cleaning 
	# all agents
	try:
		logger.info("Shutting down the root container...")
		RC.shutdown()
		logger.info("Done.")

	except Exception as e:
		logger.info("Failed to shutdown the root container")
		traceback.print_exc(file=sys.stdout)


if __name__ == '__main__':
	runContainer()
