
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


def recordAgent(agent_addr, agent_type, db_engine):
	"""
	Store the agent in DB.
	Assume that there will be only one active
	blockchain agent.
	"""

	# For now, just use the raw query
	query = "INSERT INTO `tbl_agents` (`session_id`, `agent_id`, `agent_address`, "\
			"`agent_type`, `status`, `insertion_datetime`) VALUES ('rootcontainersession', '-1', "\
			"'{}', '{}', 'active', CURRENT_TIMESTAMP)".format(agent_addr, agent_type)

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

	query = "UPDATE `tbl_agents` set `status`='dead' WHERE agent_address = '{}' "\
			"AND agent_type='{}' AND `status`='active'".format(agent_addr, agent_type)

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
	db_engine = CF.get_db_engine()

	# Initiate the blockchain agent
	blockChainAgent = BlockchainAgent(container=RC, )

	# Dump the blockchain agent address
	logger.info("Blcokchain agent initiated at {}".format(blockChainAgent.addr))

	# Record this agent to DB
	status = recordAgent(agent_addr=blockChainAgent.addr, agent_type='blockchain', db_engine=db_engine)


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
