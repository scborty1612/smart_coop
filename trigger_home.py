"""
This module triggers a home agent
provided by the home ID.

The home agent must be located into a container
and ready to be kicked in!

"""
import aiomas
import sys, traceback
from configure import Configure as CF
from sqlalchemy import create_engine
import pandas as pd

# Logging stuffs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TriggerAgent(aiomas.Agent):
	def __init__(self, container):
		super().__init__(container)

	async def run(self, agent_addr):
		home_agent = await self.container.connect(agent_addr)
		logging.info(home_agent)
		ret = await home_agent.trigger(agent_type="TRIGGER_AGENT")
		logging.info(ret)


def getAgentAddress(agent_id=None, session_id=None, db_engine=None):
	"""
	Retrieve the home agent's address given the
	session id and agent id
	"""
	# Statement to retreive agent address
	query = "SELECT `agent_address` FROM `tbl_agents` WHERE `agent_id`={} AND session_id='{}'".format(agent_id, session_id)

	# Dump the reseult into a dataframe
	df = pd.read_sql(query, db_engine)

	if len(df) < 1:
		return None

	return df['agent_address'][0]

def main():
	"""
	"""
	# Check the length of the command line arguments
	if len(sys.argv) != 3:
		logging.info("Wrong number of arguments.")
		logging.info("Usage: python trigger_home.py <session_id> <home_id>")

	# Session and agent(home) id
	session_id = sys.argv[1]
	agent_id = int(sys.argv[2])

	# Create DB engine
	# db_engine = create_engine("mysql+pymysql://{}@{}/{}".format(CF.DB_USER, CF.DB_HOST, CF.DB_NAME))
	db_engine = CF.get_db_engine()

	agent_addr = getAgentAddress(agent_id=agent_id, session_id=session_id, db_engine=db_engine)
	
	if agent_addr is None:
		logging.info("Agent address couldn't be retreived.")
		return

	logging.info(agent_addr)

	try:
		c = aiomas.Container.create(('localhost', 5560))

		trigger_agent = TriggerAgent(container=c)
		aiomas.run(until=trigger_agent.run(agent_addr))	

	except Exception as e:
		logger.info("Failed to open/create container or run the triggering agent(s)")
		traceback.print_exc(file=sys.stdout)
	finally:
		# Shutting down the container
		logger.info("Shutting down the triggering agents container.")
		c.shutdown()


if __name__ == '__main__':
	main()

