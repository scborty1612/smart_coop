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
	def __init__(self, container, bc_addr):
		super().__init__(container)

		# Store system grid exchange ()
		self.__system_grid_transfer = None

		# Store the system imbalance
		self.__system_imbalance = None

		# Address of the blcokchain agent
		self.__bc_addr = bc_addr


	@aiomas.expose
	async def observeSystemState(self,):
		"""
		"""
		bc_agent = await self.container.connect(self.__bc_addr)

		if not bc_agent:
			return False

		# Initiate the plotting
		fig = plt.figure(figsize=(15, 12))

		# =(12, 8)

		# Scan the system status periodically
		datetime_fmt = "%Y-%m-%d %H:%M:%S"

		# Fetch the currrent time from Clock
		current_datetime = self.container.clock.utcnow().format("YYYY-MM-DD HH:mm:ss")

		# Run a simulation till a specific period
		sim_end_datetime = datetime.datetime.strptime(CF.SIM_END_DATETIME, datetime_fmt)

		initiate_figure = False

		# Make sure to trigger the home agent before the simulation date is over
		while datetime.datetime.strptime(current_datetime, datetime_fmt) < sim_end_datetime:

			logging.info("Current datetime {}".format(current_datetime))

			grid_transfer, agent_list, actual_data, pred_data = await bc_agent.provideStaticSystemStatus(start_datetime=CF.SIM_START_DATETIME, 
																									 end_datetime=str(current_datetime))

			if not grid_transfer:
				continue

			# Decompose the result data structure
			self.__system_grid_transfer = pd.read_json(grid_transfer)
			# self.__system_imbalance = pd.read_json(grid_transfer)

			if len(self.__system_grid_transfer) <= 0:
				continue

			if not initiate_figure:
				
				gs = gridspec.GridSpec(nrows=len(agent_list)+1, ncols=1)
				
				# 1st axis is for system grid_transfer
				ax1 = fig.add_subplot(gs[0,:])

				# From 2nd and so on for each agent/home
				axes = []
				for i, agent in enumerate(agent_list):
					axes.append(fig.add_subplot(gs[i+1, :]))
				
				plt.ion()
				
			# Plotting the current system grid_transfer
			self.__system_grid_transfer.plot(ax=ax1, color='g', legend=not initiate_figure,)

			for i, agent in enumerate(agent_list):
				try:
					# Decompose the actual/realized data
					_actual = pd.read_json(actual_data[i])
					_actual.plot(ax=axes[i], colormap='BrBG', legend=not initiate_figure,)

					# # Decompose the predicted data
					# _prediction = pd.read_json(pred_data[i])
					# _prediction.plot(ax=axes[i], color='r', legend=not initiate_figure, figsize=figsize)

				except Exception as e:
					continue

			
			initiate_figure = True
			# self.__system_imbalance.plot(ax=ax2, color='g')
			# ax1.legend_.remove()
			# ax2.legend_.remove()
			
			# Sleep a bit before rendering the next set of data
			plt.pause(CF.DELAY)

			# Increment the clock to next period (dt=15min)
			self.container.clock.set_time(self.container.clock.time() + (1*60*CF.GRANULARITY))

			current_datetime = self.container.clock.utcnow().format("YYYY-MM-DD HH:mm:ss")

		# plt.show()

	def __plotSystemImbalance(self):
		"""
		Plotting system imbalance
		"""
		self.__system_grid_transfer.plot(figsize=(14, 6))
		plt.show()
