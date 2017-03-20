"""
Blockchain observer agent.
This agent basically tries to catch
current status of the block chain agent
and creates or publish some views/plots.
"""

# Import the agent package
import aiomas
import asyncio

# For storing and manipulating data
import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

import sys
sys.path.append("../")
from configure import Configure as CF
from util import DBGateway as DB


# Logging stuffs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys, traceback
# Define the generic agent
"""
Later, heterogenious agent will be added 
"""
class BlockchainObserver(aiomas.Agent):
	"""
	The residential agents
	"""
	def __init__(self, container, SPAgentAddr):
		super().__init__(container)

		# Record SPAgent's address
		self.__spa_addr = SPAgentAddr

		# Store system grid exchange ()
		self.__system_grid_transfer = None

		# Store the system imbalance
		self.__system_imbalance = None


	async def register(self):
		# Connect to the SP Agent
		spa_agent = await self.container.connect(self.__spa_addr,)

		# Get the alive session ID
		self.session_id = await spa_agent.getAliveSessionID()
		logging.info("session ID: {}".format(self.session_id))

		# Register the agent
		status = await spa_agent.recordAgent(session_id=str(self.session_id),
					  container_name='rootcontainer',
					  container_address=self.container._base_url,
					  agent_id=-3,
					  agent_address=self.addr,
					  agent_type='blockchain_observer',
					  agent_functionality='Blockchain Observer')
		if not status:
			logging.info("Could not register Blockchain Observer agent.")

		# Now bind the blockchain agent
		bc_address = await spa_agent.getAliveBlockchain(self.session_id)

		if not bc_address:
			raise Exception("BC agent not found!")

		# Bind with blockchain address
		self.__bc_addr = bc_address

		return True


	def __simulationTime(self,):
		# Scan the system status periodically
		datetime_fmt = "%Y-%m-%d %H:%M:%S"

		# Fetch the currrent time from Clock
		current_datetime = self.container.clock.utcnow().format("YYYY-MM-DD HH:mm:ss")

		# Run a simulation till a specific period
		sim_end_datetime = datetime.datetime.strptime(CF.SIM_END_DATETIME, datetime_fmt)
		
		# Make sure to trigger the home agent before the simulation date is over
		while datetime.datetime.strptime(current_datetime, datetime_fmt) < sim_end_datetime:
			yield current_datetime

			# Increment the clock to next period (dt=15min)
			self.container.clock.set_time(self.container.clock.time() + (1*60*CF.GRANULARITY))

			current_datetime = self.container.clock.utcnow().format("YYYY-MM-DD HH:mm:ss")


	@aiomas.expose
	async def observeSystemState(self,):
		"""
		"""
		bc_agent = await self.container.connect(self.__bc_addr)

		if not bc_agent:
			return False

		# Initiate the plotting
		fig = plt.figure(figsize=(15, 12))

		initiate_figure = False

		# Make sure to trigger the home agent before the simulation date is over
		# while datetime.datetime.strptime(current_datetime, datetime_fmt) < sim_end_datetime:
		for current_datetime in self.__simulationTime():

			logging.info("Current datetime {}".format(current_datetime))

			grid_transfer, agent_list, actual_data, pred_data = await bc_agent.provideStaticSystemStatus(start_datetime=CF.SIM_START_DATETIME, 
																									 end_datetime=str(current_datetime))

			# grid_transfer, imbalance, agent_list, actual_data, pred_data = await bc_agent.provideStaticSystemStatus(start_datetime=CF.SIM_START_DATETIME, 
			# 																						 end_datetime=str(current_datetime))

			if not grid_transfer:# or not imbalance:
				continue

			# Decompose the result data structure
			self.__system_grid_transfer = pd.read_json(grid_transfer)
			# self.__system_imbalance = pd.read_json(imbalance)

			if len(self.__system_grid_transfer) <= 0:
				continue

			if not initiate_figure:
				
				gs = gridspec.GridSpec(nrows=len(agent_list)+1, ncols=1)
				
				# 1st axis is for system imbalance
				# ax1 = fig.add_subplot(gs[0,:])

				# 2nd axis is for system grid
				ax2 = fig.add_subplot(gs[0,:])

				# From 2nd and so on for each agent/home
				axes = []
				for i, agent in enumerate(agent_list):
					axes.append(fig.add_subplot(gs[i+1, :]))
				
				plt.ion()
				
			# Plotting the current system imbalance
			# self.__system_imbalance.plot(ax=ax1, color='g', legend=not initiate_figure,)

			# Plotting the current system grid_transfer
			self.__system_grid_transfer.plot(ax=ax2, color='g', legend=not initiate_figure, )

			# Plotting columns
			plt_cols = ['use', 'gen', 'battery_power', 'battery_energy']
			for i, agent in enumerate(agent_list):
				try:
					# Decompose the actual/realized data
					_actual = pd.read_json(actual_data[i])
					_actual[plt_cols].plot(ax=axes[i], colormap='BrBG', legend=not initiate_figure,)

				except Exception as e:
					continue

			
			initiate_figure = True
			# self.__system_imbalance.plot(ax=ax2, color='g')
			# ax1.legend_.remove()
			# ax2.legend_.remove()
			
			# Sleep a bit before rendering the next set of data
			plt.pause(CF.DELAY)


		# plt.show()

	def __plotSystemImbalance(self):
		"""
		Plotting system imbalance
		"""
		self.__system_grid_transfer.plot(figsize=(14, 6))
		plt.show()
