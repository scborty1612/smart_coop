"""
Home Agent:
The generic agent of household.
The agent basically collects historical data from database.
The database credentials are presented in the configure.py.

As a next step, we will add the battery resources.
A simple battery simulator will be added. We can start with
by giving the battery with an initial SOC.
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
class HomeAgent(aiomas.Agent):
	"""
	The residential agents
	"""
	def __init__(self, container, agent_id, spa_addr, has_battery=True):
		# Register the home agent with the container
		super().__init__(container)
		
		# Localize the agent/home ID
		self.agent_id = agent_id

		# Localize the service provider agent's address
		self.__spa_addr = spa_addr

		# Load the measurement data
		self.__super_data = DB.loadGenAndDemandDataForHome(
						agent_id=self.agent_id,
						start_datetime=str(datetime.datetime.strptime(CF.SIM_START_DATETIME, "%Y-%m-%d %H:%M:%S")-datetime.timedelta(days=60)),
						end_datetime=CF.SIM_END_DATETIME)

		# Take the slice of the super data to think of the measurement data
		self.__data = self.__super_data[CF.SIM_START_DATETIME:CF.SIM_END_DATETIME].copy()

		# Empty data structure for storing the predictions
		self.__predictions = dict()

		# System grid exchange
		self.__grid_exchange = dict()

		# Systme imbalance
		self.__system_imbalance = dict()

		# Add battery to the house
		if has_battery:			
			self.__has_battery = True
			self.__addBatteryNaiveScheduler()

		# Define and initialize the mental state of the agent
		self.__mental_state = dict({'prediction': self.__predictions, 
									'grid_exchange': self.__grid_exchange,
									'system_imbalance': self.__system_imbalance,
									'data': self.__data})
		

	def showMentalState(self):
		"""
		Dump the mental state ()
		"""
		logging.info("__________{}_______state".format(self.agent_id))
		for k, v in self.__mental_state.items():
			logging.info("State: {}".format(k))
			logging.info(v)
		logging.info("____________________")

		return

	async def registerAndBind(self, session_id):
		"""
		Registering the home agent to the Service Provider
		linked to a particular session ID
		"""
		self.session_id = session_id

		# Connect to the SP Agent
		spa_agent = await self.container.connect(self.__spa_addr,)

		# Register the agent (with associated information)
		status = await spa_agent.recordAgent(session_id=str(self.session_id),
					  container_name='homecontainer',
					  container_address=self.container._base_url,
					  agent_id=self.agent_id,
					  agent_address=self.addr,
					  agent_type='home',
					  agent_functionality='Home')
		
		if not status:
			logging.info("Could not register Home agent.")

		# Now bind the blockchain agent to which the agent will send information
		# periodically
		bc_address = await spa_agent.getAliveBlockchain(session_id)

		if not bc_address:
			raise Exception("BC agent not found!")

		self.__bc_address = bc_address

		return True

	async def kill(self):
		# connect to SP Agent
		spa_agent = await self.container.connect(self.__spa_addr)

		# Kill the agent
		status = await spa_agent.killAgent(self.agent_id)

		return status

	def __addBatteryNaiveScheduler(self):
		"""
		Add a battery to the house.
		Basically, we will create a dataframe containing empty information
		with the temporal information ranging from start to end simulation datetime
		"""
		
		# Add the battery related columns to other demand/gen data
		self.__data['battery_power'] = np.zeros(len(self.__data))
		self.__data['battery_energy'] = np.zeros(len(self.__data))
		self.__data['battery_soc'] = np.zeros(len(self.__data))

		# Load after battery, pv and generator

		# Initialize at time -1
		init_soc = 0.5
		soc = init_soc

		# In case of naive battery scheduler,
		# we can run update on the actual demand.

		# Delta T
		deltaT = CF.GRANULARITY/60

		# Initial battery state
		b_state = init_soc * CF.BATTERY_CHAR['capacity']

		# Iterate over the periods 
		for i in range(len(self.__data)):
			# Current load
			c_load   = self.__data.iloc[i]['load']

			# In case of PV is higher than the load
			if c_load < 0 and soc <= CF.BATTERY_CHAR['soc_high']:
				# the battery will be charged 
				# charging amount
				charge = min(CF.BATTERY_CHAR['c_rating'], abs(c_load))
					
				# Potential state of charge
				b_state += CF.BATTERY_CHAR['c_eff'] * (abs(charge)) * deltaT
				
				# Update battery power
				self.__data.ix[i, 'battery_power'] = charge

				# Update the remaning load
				self.__data.ix[i, 'load'] += charge

				# Update the SOC
				soc = b_state/CF.BATTERY_CHAR['capacity']

			elif c_load >=0 and soc >= CF.BATTERY_CHAR['soc_low']:
				# the battery will be discharged
				discharge = min(CF.BATTERY_CHAR['d_rating'], c_load)

				# Potential state of charge
				b_state += 1/CF.BATTERY_CHAR['d_eff'] * (-discharge) * deltaT

				# Update battery power
				self.__data.ix[i, 'battery_power'] = -discharge

				# Update the load
				self.__data.ix[i, 'load'] += (-discharge) 

				# Update the SOC
				soc = b_state/CF.BATTERY_CHAR['capacity']

							
			# Populate the normalized soc
			self.__data.ix[i, 'battery_soc'] = soc

		# Just keep the battery energy also
		self.__data['battery_energy'] = self.__data['battery_soc'] * CF.BATTERY_CHAR['capacity']

		return

	def __scheduleTasks(self):
		"""
		Warning: Unused!!
		"""
		this_clock = self.container.clock
		gran_sec = CF.GRANULARITY * 60

		task = aiomas.create_task(self.__communicateBlockchain)
		this_clock.call_in(1 * gran_sec, task, '2015-01-15 00:15:00', None, None, 'UPDATE_ACTUAL')

		# this_clock.call_in(2 * gran_sec, self.__communicateBlockchain, self.__bc_address, '2015-01-15 00:30:00', None, None, 'UPDATE_ACTUAL')
		# this_clock.call_in(3 * gran_sec, self.__communicateBlockchain, self.__bc_address, None, '2015-01-15 00:15:00', '2015-01-15 00:45:00', 'RETRIEVE_IMBALANCE')

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

	@aiomas.expose
	async def triggerBlockchainCommunication(self, agent_type=None, ):
		"""
		Initiated by the trigger agent
		"""
		if agent_type != "TRIGGER_AGENT":
			return False

		# taskQued = self.__scheduleTasks()

		datetime_fmt = "%Y-%m-%d %H:%M:%S"

		# Fetch the currrent time from Clock
		current_datetime = self.container.clock.utcnow().format("YYYY-MM-DD HH:mm:ss")

		# Run a simulation till a specific period
		sim_end_datetime = datetime.datetime.strptime(CF.SIM_END_DATETIME, datetime_fmt)

		# Make sure to trigger the home agent before the simulation date is over

		while datetime.datetime.strptime(current_datetime, datetime_fmt) < sim_end_datetime:
			"""
			For now, update the actual data in every 15 mins
			and update the prediction in every 6 hours.

			Moreover, scan the blockchain for total imbalance
			"""
			logging.info("{}. Current datetime {}".format(self.agent_id, current_datetime))
			
			await self.__communicateBlockchain(current_datetime=str(current_datetime), 
				mode='UPDATE_ACTUAL')

			# Check whether its the time to predict some data
			dt_current = datetime.datetime.strptime(current_datetime, datetime_fmt)
			c_hour = int(dt_current.strftime("%H"))
			c_min = int(dt_current.strftime("%M"))

			if c_min == 0 and (c_hour%6) == 0:
				# time for predict
				logging.info("Predict something at {}".format(str(dt_current)))

				self.__loadPrediction = self.__getLoadPredictionLocal(starting_datetime=dt_current)
				
				# Just for the sake of re-usability
				# store the prediction to a dictionary
				self.__predictions.update({str(dt_current): self.__loadPrediction})

				# Now, communicate with the blockchain to update the prediction
				status = await self.__communicateBlockchain(mode='UPDATE_PREDICTION')

			# Get the system imbalance every hour
			if c_min == 30:
				# self.showMentalState()

				logging.info("{}. Time for fetching energy exchange infor from BC...".format(self.agent_id))

				# Communicating with BC
				grid_exchange = await self.__communicateBlockchain( 
					start_datetime=CF.SIM_START_DATETIME, end_datetime=str(dt_current), 
					mode='RETRIEVE_GRID_EXCHANGE')

				# Handling the returned dataframe of overall imbalance
				if grid_exchange is not None:

					# Store the system imbalance
					self.__grid_exchange.update({str(dt_current): pd.read_json(grid_exchange)})
					# logging.info("Current system imbalance {}".format(self.__sys_imbalance[str(dt_current)]))

			# if c_min == 30 and c_hour%3 == 0:
			# 	logging.info("{}. Time for fetching System Imbalance infor from BC...".format(self.agent_id))

			# 	# Communicating with BC
			# 	system_imbalance = await self.__communicateBlockchain( 
			# 		start_datetime=CF.SIM_START_DATETIME, end_datetime=str(dt_current), 
			# 		mode='RETRIEVE_SYSTEM_IMBALANCE')

			# 	# Handling the returned dataframe of overall imbalance
			# 	if system_imbalance is not None:

			# 		# Store the system imbalance
			# 		self.__system_imbalance.update({str(dt_current): pd.read_json(system_imbalance)})
			# 		# logging.info("Current system imbalance {}".format(self.__sys_imbalance[str(dt_current)]))

			# Wait for a moment
			await asyncio.sleep(CF.DELAY)

			# Increment the clock to next period (dt=15min)
			self.container.clock.set_time(self.container.clock.time() + (1*60*CF.GRANULARITY))

			current_datetime = self.container.clock.utcnow().format("YYYY-MM-DD HH:mm:ss")
			# current_datetime = current_datetime + datetime.timedelta(minutes=15)

		
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
		lp = LoadPredictor()

		# Relay the input (for now, only historicla data)
		lp.input(_type='historic_data', _data=self.__super_data[:starting_datetime]['use'])

		# Perform the prediction (training the models will be done inside the .predict method)
		prediction_df = lp.predict(t=str(starting_datetime), window=prediction_window)

		# Just for plotting
		actual_df = self.__super_data[starting_datetime:str(starting_datetime+datetime.timedelta(minutes=(prediction_window)*CF.GRANULARITY))]['use']

		_, ax = plt.subplots()
		plt.plot(np.array(prediction_df), label="Prediction")
		plt.plot(np.array(actual_df), label="Actual")
		plt.legend()
		plt.show()
		
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
