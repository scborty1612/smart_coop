
"""
This module contains the Root Container that hosts
"Blockchain" agent and "Blockchain observer" agent.
"""
import aiomas
from agents.BlockchainAgent import BlockchainAgent
from agents.BlockchainObserver import BlockchainObserver
from agents.ServiceProviderAgent import ServiceProviderAgent

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
	# Create the root container
	RC = aiomas.Container.create(('localhost', 5555), clock=CLOCK)

	# Create a Service Provider agent that basically
	# registers every agent that will be created in root container and 
	# in the other containers
	SPAgent = ServiceProviderAgent(container=RC)

	# Initiate the blockchain agent
	blockChainAgent = BlockchainAgent(container=RC, SPAgentAddr=SPAgent.addr)

	# Record the blockchain agent
	aiomas.run(until=blockChainAgent.register())

	# Initiate the blockchain observer agent
	blockChainObserver = BlockchainObserver(container=RC, SPAgentAddr=SPAgent.addr) 

	# Record the blockchain observer agent
	aiomas.run(until=blockChainObserver.register())

	# Bind the blockchain agent with blockchain observer
	# Run the event loop
	try:
		logger.info("Running the event loop. The blockchain agent is open to be connected!")
		logger.info("Please use session id:{}".format(SPAgent.session_id))

		aiomas.run()
	
	except KeyboardInterrupt:
		logging.info("Keyboard Interrupted")
	except Exception as e:
		traceback.print_exc(file=sys.stdout)
	
	try:
		logger.info("Shutting down the root container...")
		RC.shutdown()
		logger.info("Done.")		
		
		logger.info("Killing the current blockchain, observer and service provider agents")

		# Killing blockchain observer
		aiomas.run(until=SPAgent.killAgent(agent_id=-3))

		# Killing blockchain
		aiomas.run(until=SPAgent.killAgent(agent_id=-2))

		# Killing the service provider agent itself
		aiomas.run(until=SPAgent.killAgent(agent_id=-1))

	except Exception as e:
		logger.info("Failed to shutdown the root container")
		traceback.print_exc(file=sys.stdout)


if __name__ == '__main__':
	runContainer()
