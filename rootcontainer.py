
"""
This module contains the Root Container that hosts
"Blockchain" agent and "Blockchain observer" agent.
"""
import aiomas
from agents.BlockchainAgent import BlockchainAgent
from agents.BlockchainObserver import BlockchainObserver

from agents.HomeAgent import HomeAgent
from configure import Configure as CF
from util import DBGateway as DB

from sqlalchemy import create_engine
import sys, traceback

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agree on clocking
CLOCK = aiomas.ExternalClock(CF.SIM_START_DATETIME)


def runContainer():	
	# Creating container contains the home agents
	# and prediction agents
	RC = aiomas.Container.create(('localhost', 5555), clock=CLOCK)

	# Initiate the blockchain agent
	blockChainAgent = BlockchainAgent(container=RC)

	# Record this agent to DB
	status = DB.recordAgent(agent_addr=blockChainAgent.addr, agent_type='blockchain',)

	# Initiate the blochain observer agent
	blockChainObserver = BlockchainObserver(container=RC, bc_addr=blockChainAgent.addr) 

	# Record this agent to DB
	status = DB.recordAgent(agent_addr=blockChainObserver.addr, agent_type='blockchain_observer', agent_id=-2)

	# Bind the blockchain agent with blockchain observer

	# Run the event loop
	try:
		logger.info("Running the event loop. The blockchain agent is open to be connected!")
		aiomas.run()
	
	except KeyboardInterrupt:
		logging.info("Keyboard Interrupted")
	except Exception as e:
		traceback.print_exc(file=sys.stdout)
	
	# Now run the blockchain observer agent
	# try:
	# 	aiomas.run(until=blockChainObserver.observeSystemState())
	# except Exception as e:
	# 	traceback.print_exc(file=sys.stdout)

	# Shutting donw the controller and thereby cleaning clearing all the live agent
	# under this container.
	try:
		logger.info("Shutting down the root container...")
		RC.shutdown()
		logger.info("Done.")		
		
		logger.info("Killing the current blockchain and observer agents")
		status = DB.killAgent(agent_addr=blockChainAgent.addr, agent_type='blockchain')
		status = DB.killAgent(agent_addr=blockChainObserver.addr, agent_type='blockchain_observer')


		if status:
			logger.info("Done.")
		else:
			logger.info("Couldnot kill the agent!")

	except Exception as e:
		logger.info("Failed to shutdown the root container")
		traceback.print_exc(file=sys.stdout)


if __name__ == '__main__':
	runContainer()
