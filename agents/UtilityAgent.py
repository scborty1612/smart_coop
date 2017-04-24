"""
Utility Agent
"""

# Import the agent package
import aiomas
import asyncio

# For storing and manipulating data
import pandas as pd
import numpy as np
import datetime
import arrow

import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from configure import Configure as CF
from util import DBGateway as DB
from util import HomeList as HL
# Load the local predictor
# This import will be removed when we are going to use the predictor
# as a remote process
from process.LoadPredictor import LoadPredictor
from process.Scheduler import SchedulerCombined


# Import the required processes
# from process.LoadPredictor import LoadPredictor
import zmq
import json

# Logging stuffs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

"""
The generic home agent
"""
class UtilityAgent(aiomas.Agent):
	"""
	The residential agents
	"""
	def __init__(self, container, SPAgentAddr):
		# Register the agent with the container
		super().__init__(container)
		
		# Record SPAgent's address
		self.__spa_addr = SPAgentAddr

		# save locally the agent information that will be required to 
		# to perform the combined scheduling
		self.__agent_info = dict()
		
		# Window of operation (i.e. window of prediction and scheduling optimization)
		self.__window = 96

	async def register(self):
		# Connect to the SP Agent
		self.__spa_agent = await self.container.connect(self.__spa_addr,)

		# Get the alive session ID
		self.session_id = await self.__spa_agent.getAliveSessionID()
		logging.info("session ID: {}".format(self.session_id))

		# Register the agent
		status = await self.__spa_agent.recordAgent(session_id=str(self.session_id),
					  container_name='rootcontainer',
					  container_address=self.container._base_url,
					  agent_id=-4,
					  agent_address=self.addr,
					  agent_type='utility',
					  agent_functionality='Utility')
		if not status:
			logging.info("Could not register Utility agent.")

		# Now bind the blockchain agent
		bc_address = await self.__spa_agent.getAliveBlockchain(self.session_id)

		if not bc_address:
			raise Exception("Utility not found!")

		# Bind with blockchain address
		self.__bc_address = bc_address

		return True


	def __simulationTime(self,):
		# Scan the system status periodically
		datetime_fmt = "%Y-%m-%d %H:%M:%S"

		# Fetch the currrent time from Clock
		# current_datetime = self.container.clock.utcnow().format("YYYY-MM-DD HH:mm:ss")

		# For now, use different clock
		current_datetime = datetime.datetime.strptime(CF.SIM_START_DATETIME, datetime_fmt)

		# Run a simulation till a specific period
		sim_end_datetime = datetime.datetime.strptime(CF.SIM_END_DATETIME, datetime_fmt)
		
		# Make sure to trigger the home agent before the simulation date is over
		# while datetime.datetime.strptime(current_datetime, datetime_fmt) < sim_end_datetime:
		while True:
			yield current_datetime

			# # Increment the clock to next period (dt=15min)
			# self.container.clock.set_time(self.container.clock.time() + (1*60*CF.GRANULARITY))

			# current_datetime = self.container.clock.utcnow().format("YYYY-MM-DD HH:mm:ss")
			current_datetime = current_datetime + datetime.timedelta(minutes=CF.GRANULARITY)
	
	@aiomas.expose
	async def energyExchange(self,):
		"""
		Initiate the energy exchange between prosumers
		"""

	@aiomas.expose
	async def triggerBlockchainCommunication(self, agent_type=None, ):
		"""
		Initiated by the trigger agent
		"""
		if agent_type != "TRIGGER_AGENT":
			return False

		# List the active agents
		# Later we will take it from Service Provider
		homes_address = dict()
		for agent_id in HL.get():
			add = await self.__spa_agent.getAliveAgentAddress(self.session_id, agent_id)
			homes_address.update({agent_id: add})

		# Get the BC agent
		bc_agent = await self.container.connect(self.__bc_address)

		datetime_fmt = "%Y-%m-%d %H:%M:%S"

		# Fetch the currrent time from Clock
		current_datetime = self.container.clock.utcnow().format("YYYY-MM-DD HH:mm:ss")

		# Run a simulation till a specific period
		sim_end_datetime = datetime.datetime.strptime(CF.SIM_END_DATETIME, datetime_fmt)

		# Make sure to trigger the home agent before the simulation date is over
		for current_datetime in self.__simulationTime():
			"""
			For now, update the actual data in every 15 mins
			and update the prediction in every 6 hours.

			Moreover, scan the blockchain for total imbalance
			"""
			logging.info("{}. Current datetime {}".format(self.__class__.__name__, current_datetime))


			# Check whether its the time to predict some data
			# dt_current = datetime.datetime.strptime(current_datetime, datetime_fmt)
			dt_current = current_datetime
			c_hour = int(dt_current.strftime("%H"))
			c_min = int(dt_current.strftime("%M"))

			# Scheduling time
			# if c_min == 0 and c_hour in [0, 12, ]:
			# 	logging.info("{}: Scheduling time".format(current_datetime))

			# 	for agent_id, agent_address in homes_address.items():
			# 		home_agent = await self.container.connect(agent_address)

			# 		# Go for scheduling using battery
			# 		info = await home_agent.scheduleBattery(at=str(dt_current))

			# 		if info is None:
			# 			logging.info("Nothing to schedule, the agent may not have a battery")
			# 		else:
			# 			logging.info("Broadcasting schedule to BC")

			if c_min == 30:
				
				logging.info("Get actual data at {}".format(str(dt_current)))

				# Every 2 hours
				# Fetch the realized demand from all agent
				for agent_id, agent_address in homes_address.items():
					# Connect with the agent
					home_agent = await self.container.connect(agent_address)

					# Retrieve the information
					info = await home_agent.provideRealizedData(till=str(current_datetime))

					# Push it to BC
					await bc_agent.updateActualData(agent_id=agent_id, data_serialized=info)


			# if c_min == 15 and c_hour in [0]: # Open-loop solution
			if True: # Closed-loop
				# Basically, perform it at every period
				# ask for prediction and plan at the first hour
				logging.info("Predict at {}".format(str(dt_current)))

				# Fetch the predictions
				for agent_id, agent_address in homes_address.items():
					# Connect with the agent
					home_agent = await self.container.connect(agent_address)

					# Retrieve the information
					demand_p, gen_p = await home_agent.proivdePredictedData(since=str(current_datetime), period=self.__window)

					# Push it to BC
					await bc_agent.updatePredictedData(agent_id=agent_id, data_serialized=demand_p)

					# Store the predictions to agent's information
					this_agent_info = dict({'demand_p': pd.read_json(demand_p),
											'gen_p': pd.read_json(gen_p)})

					# Retreive the battery information
					battery_info = await home_agent.provideBatteryInfo(this_datetime=str(dt_current))
					this_agent_info.update({'battery_info': battery_info})

					# Store locally
					self.__agent_info.update({agent_id: this_agent_info})


				# Now go for scheduling
				# logging.info(self.__agent_info)
				opt_b_power = self.__scheduleAndExchange()
				logging.info(opt_b_power)

				# # Now toss over the agent specific optimized battery power
				for agent_id, b_power in opt_b_power.items():
					# connect with the agent
					home_agent = await self.container.connect(homes_address[agent_id])

					# Deliver the optimized power for the current time
					await home_agent.deployBattery(this_datetime=str(dt_current), battery_power=b_power[1])
					# await home_agent.deployBatteryOpen(this_datetime=str(dt_current), battery_powers=list(b_power[1:]))


			if c_hour == 23 and c_min >= 45:
				# Time to write down the agent specific optimization result

				for agent_id, agent_address in homes_address.items():
					# Connect with the agent
					logging.info("Writing down for Agent {}".format(agent_id))

					home_agent = await self.container.connect(agent_address)

					await home_agent.writeOptResult(this_datetime=str(dt_current))


			# Wait for a moment
			await asyncio.sleep(CF.DELAY)

			# # Increment the clock to next period (dt=15min)
			# self.container.clock.set_time(self.container.clock.time() + (1*60*CF.GRANULARITY))

			# current_datetime = self.container.clock.utcnow().format("YYYY-MM-DD HH:mm:ss")
			# # current_datetime = current_datetime + datetime.timedelta(minutes=15)

		
		return True

	def __scheduleAndExchange(self,):
		"""
		Scheduling method
		"""
		scheduler = SchedulerCombined(agent_info=self.__agent_info, granular=15, periods=self.__window)
		b_power, b_status = scheduler.optimize()

		# scheduler.plotCurrentResult()

		return b_power

