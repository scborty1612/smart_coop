
"""
This module contains the Root Container that hosts
"Blockchain" agent.

"""
import aiomas
from agents.BlockchainAgent import BlockchainAgent
from agents.BlockchainObserver import BlockchainObserver

from agents.HomeAgent import HomeAgent
from configure import Configure as CF

from sqlalchemy import create_engine
import sys, traceback

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recordAgent(agent_addr, agent_type, db_engine):
	"""
	Store the agent in DB.
	Assume that there will be only one active
	blockchain agent.
	"""

	# For now, just use the raw query
	query = "INSERT INTO `{}` (`session_id`, `agent_id`, `agent_address`, "\
			"`agent_type`, `status`, `insertion_datetime`) VALUES ('rootcontainersession', '-1', "\
			"'{}', '{}', 'active', CURRENT_TIMESTAMP)".format(CF.TBL_AGENTS, agent_addr, agent_type)

	# Execute the query
	try:
		result = db_engine.execute(query) 
	except Exception as e:
		logger.info("Query failed to execute!")
		traceback.print_exc(file=sys.stdout)
		return False

	return True	

def killAgent(agent_addr, agent_type, db_engine):
	"""
	Kill any agent by changing the status
	to 'dead'
	"""

	query = "UPDATE `{}` set `status`='dead' WHERE agent_address = '{}' "\
			"AND agent_type='{}' AND `status`='active'".format(CF.TBL_AGENTS, agent_addr, agent_type)

	# logging.info(query)

	try:
		db_engine.execute(query)
	except Exception as e:
		logger.info("Query failed to execute!")
		traceback.print_exc(file=sys.stdout)
		return False

	return True	

def runContainer():	
	# Creating container contains the home agents
	# and prediction agents
	RC = aiomas.Container.create(('localhost', 5555))

	# Create the DB engine
	# db_engine = create_engine("mysql+pymysql://{}@{}/{}".format(CF.DB_USER, CF.DB_HOST, CF.DB_NAME))
	try:
		db_engine = CF.get_db_engine()
	except Exception as e:
		logger.info("Can't connect to DB.")
		return 		

	# Initiate the blockchain agent
	blockChainAgent = BlockchainAgent(container=RC)

	# Record this agent to DB
	status = recordAgent(agent_addr=blockChainAgent.addr, agent_type='blockchain', db_engine=db_engine)

	# Initiate the blochain observer agent
	blockChainObserver = BlockchainObserver(container=RC) 

	# Record this agent to DB
	status = recordAgent(agent_addr=blockChainObserver.addr, agent_type='blockchain_observer', db_engine=db_engine)

	# Dump the blockchain agent address
	logger.info("Blcokchain agent initiated at {}".format(blockChainAgent.addr))
	logger.info("Blcokchainobserver agent initiated at {}".format(blockChainObserver.addr))

	# Run the event loop
	try:
		logger.info("Running the event loop. The blockchain agent is open to be connected!")
		aiomas.run()
	
	except KeyboardInterrupt:
		logging.info("Keyboard Interrupted")
		# Just run the blockchain observer agent
		# yea...its a bad design, just bear with me
		aiomas.run(until=blockChainObserver.observeSystemImbalance(blockChainAgent.addr))

	except Exception as e:
		traceback.print_exc(file=sys.stdout)
	# Shutting donw the controller and thereby cleaning 
	# all agents
	try:
		logger.info("Shutting down the root container...")
		RC.shutdown()
		logger.info("Done.")
		
		logger.info("Killing Blockchain agent")
		status = killAgent(agent_addr=blockChainAgent.addr, agent_type='blockchain', db_engine=db_engine)

		if status:
			logger.info("Done.")
		else:
			logger.info("Couldnot kill the agent!")

	except Exception as e:
		logger.info("Failed to shutdown the root container")
		traceback.print_exc(file=sys.stdout)


if __name__ == '__main__':
	runContainer()
