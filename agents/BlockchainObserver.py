"""
Blockchain observer agent.
This agent basically tries to catch
current status of the blockchain agent
and creates or publish some views/plots.
"""

# Import the agent package
import aiomas
import asyncio

# For storing and manipulating data
import pandas as pd
import numpy as np
import datetime
import time
import math

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
		# while datetime.datetime.strptime(current_datetime, datetime_fmt) < sim_end_datetime:
		while True:
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

		# Flag to check whethere the figure is initiated
		initiate_figure = False

		# Previous list of agents
		prev_no_agents = 0

		# Make sure to trigger the home agent before the simulation date is over
		# while datetime.datetime.strptime(current_datetime, datetime_fmt) < sim_end_datetime:
		for current_datetime in self.__simulationTime():

			logging.info("{}. Current datetime {}".format(self.__class__.__name__, current_datetime))
			# Updates on energy exchange and system imbalance

			try:
				grid_transfer, imbalance, agent_list, actual_data, pred_data = \
				await bc_agent.provideSystemStatus(start_datetime=CF.SIM_START_DATETIME, 
												end_datetime=str(current_datetime))
			except Exception as e:
				logging.info("No data received yet!")
				time.sleep(CF.DELAY)
				continue

			if not grid_transfer or not imbalance:
				# Delay a bit
				# await asyncio.sleep(CF.DELAY)				
				time.sleep(CF.DELAY)
				continue

			# Decompose the result data structure
			self.__system_grid_transfer = pd.read_json(grid_transfer)
			self.__system_imbalance = pd.read_json(imbalance)

			print(self.__system_imbalance)
			print(self.__system_grid_transfer)


			if len(self.__system_grid_transfer) <= 0 or len(self.__system_imbalance) <= 0:
				# Delay a bit
				# await asyncio.sleep(CF.DELAY)				
				time.sleep(CF.DELAY)
				continue

			max_rows = 3

			if not initiate_figure:
				# Initiate the plotting
				fig = plt.figure(figsize=(18, 12))
				
				nrows = max_rows + 1
				# ncols = len(agent_list)//max_rows + 1
				ncols = int(math.ceil(len(agent_list)/max_rows))

				gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
				
				# 1st axis is for system imbalance and grid exchange
				ax = fig.add_subplot(gs[0,:])

				# From 2nd and so on for each agent/home
				axes = []
				for i, agent in enumerate(agent_list):
					_r = i//ncols+1
					_c = int(i%ncols)
					# axes.append(fig.add_subplot(gs[i+1, :]))
					axes.append(fig.add_subplot(gs[_r,_c]))
				
				gs.tight_layout(fig, pad=3.5)
				plt.ion()
				
			"""Plotting the current system imbalance and grid exchange"""

			# Clearing the current line is required since the arrival of predictions 
			# is asynchronous in Blockchain
			total_lines = len(ax.lines)
			for i in range(total_lines):
				if ax.lines[0]:
					del ax.lines[0]

			self.__system_imbalance.plot(ax=ax, color='b', legend=not initiate_figure,)
			# self.__system_grid_transfer.plot(ax=ax, color='g', legend=not initiate_figure, )

			# Plotting columns
			plt_cols = ['use', 'gen', 'battery_power', 'battery_energy']

			# Column(s) that will be plotted as bar
			bar_cols = ['battery_power']

			# Plotting colors
			plt_colors = ['#3778bf', '#5e819d', '#7bb274', '#825f87']

			# Plot each agent's information
			for i, agent in enumerate(agent_list):
				
				# Clearing the lines 
				total_lines = len(axes[i].lines)

				for x in range(total_lines):
					if axes[i].lines[0]:
						del axes[i].lines[0]			

				try:
					# Decompose the actual/realized data
					_actual = pd.read_json(actual_data[i])

					# Before plotting the columns, make sure they exist
					for _i, col in enumerate(plt_cols):
						if col not in _actual.columns.values.tolist(): continue					
						
						if col in bar_cols:
							# For now, just change the alpha
							_actual[col].plot(ax=axes[i], color=plt_colors[_i], alpha=0.5,
											   legend=not initiate_figure, title="Agent: {}".format(agent))
						else:							
							_actual[col].plot(ax=axes[i], color=plt_colors[_i], 
											   legend=not initiate_figure, title="Agent: {}".format(agent))

				# If something wrong, move ahead
				except Exception as e:
					continue

			
			initiate_figure = True
			
			# Sleep a bit (more than the agents' clock) 
			# before rendering the next set of data
			plt.pause(CF.DELAY)

		# plt.show()

	@aiomas.expose
	async def observeSystemStateWoutPlot(self,):
		"""
		"""
		bc_agent = await self.container.connect(self.__bc_addr)

		if not bc_agent:
			return False

		# Retrive the current status
		grid_transfer, imbalance, agent_list, actual_data, pred_data = await bc_agent.provideSystemStatus(start_datetime=CF.SIM_START_DATETIME, 
																									 end_datetime=CF.SIM_END_DATETIME)
		# Decompose the result data structure
		self.__system_grid_transfer = pd.read_json(grid_transfer)
		self.__system_imbalance = pd.read_json(imbalance)

		# dump them
		# logging.info(self.__system_imbalance)
		# Check the histograms
		_, ax = plt.subplots()
		_stds = []
		_means = []
		for hour in range(24):
			_means.append(np.mean(self.__system_imbalance[self.__system_imbalance.index.hour == hour]))
			_stds.append(np.std(self.__system_imbalance[self.__system_imbalance.index.hour == hour]))
		
		plt.plot(_means, label='mean')
		plt.plot(_stds, label='stds')
		plt.xlabel("Hour")
		plt.ylabel("kWh")
		plt.legend()
		plt.show()

		return True



	def __plotSystemImbalance(self):
		"""
		Plotting system imbalance
		"""
		self.__system_grid_transfer.plot(figsize=(14, 6))
		plt.show()
