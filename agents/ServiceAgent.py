"""
Home agent 1
"""

# Import the agent package
import aiomas

# Importing the database access stuffs
from sqlalchemy import create_engine

# For storing and manipulating data
import pandas as pd
import numpy as np

# Define the generic agent
"""
Later, heterogenious agent will be added 
"""
class ServicewAgent(aiomas.Agent):
	"""
	The residential agents
	"""
	def __init__(self, container, agent_id=0):
		super().__init__(container)
		self.__agentStatshes = dict()

	async def run(self, addr):
		"""
		Dropping for testing purpose.
		Will be integrated for inter-agent communications.

		"""
		remote_agent = await self.container.connect(addr)
		ret = await remote_agent.service(42)
		print("{} got {} from {}".format(self.agent_id, ret, remote_agent.agent_id))

	@aiomas.expose
	def service(self, value):
		return value

"""
Create prediction agent
"""

