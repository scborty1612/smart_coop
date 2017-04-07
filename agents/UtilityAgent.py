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



	async def __communicateBlockchain(self,
		current_datetime=None, 
		start_datetime=None, 
		end_datetime=None, 
		mode='UPDATE_ACTUAL'):
		"""
		Start communicating with Blockchain
		agent to toss over the actual or forecasted
		demand.
		"""
		bc_agent = await self.container.connect(self.__bc_address)

		if mode is 'UPDATE_ACTUAL':

			# Only provide the generation and demand data

			cdf = self.__data[str(CF.SIM_START_DATETIME): str(current_datetime)]

			if cdf is None:
				logger.info("No data is available for the timeframe [{} to {}]".format(str(CF.SIM_START_DATETIME), str(current_datetime)))
				return None

			result = await bc_agent.updateActualData(agent_id=self.agent_id, data_serialized=cdf.to_json())
			# logging.info("received {} from BC agent".format(ret))
		elif mode is 'UPDATE_PREDICTION':
			result = await bc_agent.updatePredictedData(agent_id=self.agent_id, data_serialized=self.__loadPrediction.to_json())

		elif mode is 'RETRIEVE_GRID_EXCHANGE':
			result = await bc_agent.provideGridifyEnergy(start_datetime, end_datetime)

		elif mode is 'RETRIEVE_SYSTEM_IMBALANCE':
			result = await bc_agent.provideSystemImbalance(start_datetime, end_datetime)

		else:
			logger.info("Unrecognized MODE.")
			result = None

		return result


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


			if c_min == 30:# and (c_hour%2) == 0:
				
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



			if c_min == 0:# and (c_hour%6) == 0:
				# time for predict
				logging.info("Predict at {}".format(str(dt_current)))

				# Every 6 hours
				# Fetch the predictions
				for agent_id, agent_address in homes_address.items():
					# Connect with the agent
					home_agent = await self.container.connect(agent_address)

					# Retrieve the information
					info = await home_agent.proivdePredictedData(since=str(current_datetime))

					# Push it to BC
					await bc_agent.updatePredictedData(agent_id=agent_id, data_serialized=info)


			# Wait for a moment
			await asyncio.sleep(CF.DELAY)

			# # Increment the clock to next period (dt=15min)
			# self.container.clock.set_time(self.container.clock.time() + (1*60*CF.GRANULARITY))

			# current_datetime = self.container.clock.utcnow().format("YYYY-MM-DD HH:mm:ss")
			# # current_datetime = current_datetime + datetime.timedelta(minutes=15)

		
		return True


	def receiveActualData(self, current_datetime):
		"""
		This method mimics the behavior of sensor; a kind of perceptor
		"""
		

	def __getLoadPredictionLocal(self, starting_datetime, prediction_window=96):
		"""
		Load prediction where the predictor is located locally.
		"""

		# Instantiate the predictor class
		lp = LoadPredictor(agent_id=self.agent_id)

		# Relay the input (for now, only historicla data)
		lp.input(_type='historic_data', _data=self.__super_data[:starting_datetime]['use'])

		# Perform the prediction (training the models will be done inside the .predict method)
		prediction_df = lp.predict(t=str(starting_datetime), window=prediction_window)

		# Just for plotting
		# actual_df = self.__super_data[starting_datetime:str(starting_datetime+datetime.timedelta(minutes=(prediction_window)*CF.GRANULARITY))]['use']

		# _, ax = plt.subplots()
		# plt.plot(np.array(prediction_df), label="Prediction")
		# plt.plot(np.array(actual_df), label="Actual")
		# plt.legend()
		# plt.show()
		
		return prediction_df

	def __getLoadPredictionRemote(self, starting_datetime, prediction_window=96):
		"""
		Load prediction where the predictor is located remotely.
		Get the load prediction for next `prediction_window`
		starting from `starting_datetime`
		"""

		# Prepare the historical input data prior to the start_datetime
		input_data = pd.DataFrame(np.array(self.__super_data[:starting_datetime]['use']), columns=['load'])
		input_data.index = self.__super_data[:starting_datetime]['use'].index

		# Prepare the datastructure to be sent over netowrk
		# for prediction purpose
		network_data = dict({'starting_datetime': str(starting_datetime),
							 'prediction_window': prediction_window,
							 'input_data': input_data.to_json()})

		# Load predictor host
		load_predictor_host = "tcp://127.0.0.1:9090"

		
		context = zmq.Context()
		logger.info("Connecting to the predictor server {}".format(load_predictor_host))
		socket = context.socket(zmq.REQ)
		socket.connect(load_predictor_host)

		# Sending the input data to the predictor host
		socket.send_json(json.dumps(network_data))

		# Get the reply with the prediction
		prediction_df_serialized = socket.recv_json()

		# Unserailzed
		prediction_df = pd.read_json(prediction_df_serialized)

		# Just for plotting
		actual_df = self.__super_data[starting_datetime:str(starting_datetime+datetime.timedelta(minutes=(prediction_window)*CF.GRANULARITY))]['use']

		_, ax = plt.subplots()
		plt.plot(np.array(prediction_df), label="Prediction")
		plt.plot(np.array(actual_df), label="Actual")
		plt.legend()
		plt.show()
		
		return prediction_df
