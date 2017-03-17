"""
This module triggers a home agent
provided by the home ID.

The home agent must be located into a container
and ready to be kicked in!

"""
import aiomas
import sys, traceback
from util import DBGateway as DB

from sqlalchemy import create_engine
import pandas as pd
import click

# Logging stuffs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TriggerAgent(aiomas.Agent):
	def __init__(self, container):
		super().__init__(container)

	async def triggerHomeAgent(self, agent_addr):
		logging.info("Connecting to {}".format(agent_addr))
		home_agent = await self.container.connect(agent_addr,)
		logging.info(home_agent)
		ret = await home_agent.triggerBlockchainCommunication(agent_type="TRIGGER_AGENT")
		logging.info(ret)

	async def triggerBCOAgent(self, agent_addr):
		logging.info("Connecting to {}".format(agent_addr))
		bco_agent = await self.container.connect(agent_addr,)
		logging.info(bco_agent)
		ret = await bco_agent.observeSystemState()
		logging.info(ret)


"""
CLI for triggering home agent
"""

@click.command()
@click.option('--port',  required=True,
              help="Port where the tigger agent's container is located")
@click.option('--session-id', required=True,
              help="Session ID to locate agents address")
@click.option('--agent-id',  required=True,
              help='Agent (Home) ID to be kicked in')


def main(session_id, agent_id, port):
	
	# Retrieve the agent id and agent type
	agent_addr, agent_type = DB.getAgentInfo(agent_id=int(agent_id), session_id=session_id)
	
	# If nothing found, do nothing!
	if agent_addr is None:
		logging.info("Agent address couldn't be retreived. Make sure to provide correct session ID.")
		return

	logging.info("Agent's address: {}".format(agent_addr))

	try:
		# Create the container the host trigger agent
		c = aiomas.Container.create(('localhost', int(port)))

		# Host the trigger agent
		trigger_agent = TriggerAgent(container=c)
		
		# Kick the home agent by trigger agent
		if agent_type == 'home':
			aiomas.run(until=trigger_agent.triggerHomeAgent(agent_addr))

		# or kick the observer agent to have a look at the system imbalance	
		elif agent_type == 'blockchain_observer':
			aiomas.run(until=trigger_agent.triggerBCOAgent(agent_addr))

	except OSError:
		logger.info("Probably the provided port is already in use or the agent in action is dead!")
		
	except ConnectionResetError:
		logger.info("Probably the home agent died.")

	except Exception as e:
		logger.info("Failed to open/create container or run the triggering agent!")
		traceback.print_exc(file=sys.stdout)

	# Shutting down the container
	logger.info("Shutting down the triggering agents container.")	
	try:		
		c.shutdown()
	except Exception as e:
		logger.info("Couldn't shutdown the container!")

	return

if __name__ == '__main__':
	main()
