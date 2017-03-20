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

# Logging stuffs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
The generic home agent
"""
class HomeAgent(aiomas.Agent):
	"""
	The residential agents
	"""
	def __init__(self, container, agent_id, has_battery=True):
		super().__init__(container)
		self.agent_id = agent_id
		logging.info("Agent {}. Address: {} says hi!!".format(agent_id, self))
		logging.info("Agent {} fetching data".format(agent_id))
		logging.info("")

		# Assigning the databse connectin
		# the idea is to create only a single connection
		# Let the mysql's internal connection manager handle
		# all the connection related issue
		self.__conn = DB.get_db_engine()

		# Load the measurement data
		self.__data = DB.loadGenAndDemandDataForHome(agent_id=self.agent_id,
													 start_datetime=CF.SIM_START_DATETIME,
													 end_datetime=CF.SIM_END_DATETIME)
		# Load information regarding the house
		# such as location, household devices, etc.
		# self.__house_info = self.__loadHouseInfo()

		# Empty data structure for storing the predictions
		self.__predictions = dict()

		# System grid exchange
		self.__grid_exchange = dict()

		# Systme imbalance
		self.__system_imbalance = dict()

		# Initial battery SOC
		# self.__init_soc = CF.INIT_SOC

		# Add battery to the house
		if has_battery:			
			self.__has_battery = True
			self.__addBatteryNaiveScheduler()
		
		# self.__data[['load', 'battery_power', 'battery_energy']].plot()
		# plt.show()


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

	def __loadHouseInfo(self):
		"""
		This method loads the information regarding the
		house and store them into a dataframe
		"""
		query = "SELECT * FROM {} WHERE house_id = {}".format(DB.TBL_HOUSE_INFO, self.agent_id)

		df = pd.read_sql(query, self.__conn)

		return df

	def __analyzeData(self):
		"""
		This method essentially provides some analysis on the historical
		data. How the flexibility will be calculated?
		"""
		pass


	def setBlockchainAddress(self, bc_address):
		"""
		Set the blockchain address.
		"""
		self.__bc_address = bc_address


	def __scheduleTasks(self):
		"""
		Unused!
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
				self.__loadPrediction = self.__getLoadPrediction(starting_datetime=dt_current)
				
				# Just for the sake of re-usability
				# store the prediction to a dictionary
				self.__predictions.update({str(dt_current): self.__loadPrediction})

				# Now, communicate with the blockchain to update the prediction
				status = await self.__communicateBlockchain(mode='UPDATE_PREDICTION')

			# Get the system imbalance every hour
			if c_min == 30:
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

			if c_min == 30 and c_hour%3 == 0:
				logging.info("{}. Time for fetching System Imbalance infor from BC...".format(self.agent_id))

				# Communicating with BC
				system_imbalance = await self.__communicateBlockchain( 
					start_datetime=CF.SIM_START_DATETIME, end_datetime=str(dt_current), 
					mode='RETRIEVE_SYSTEM_IMBALANCE')

				# Handling the returned dataframe of overall imbalance
				if system_imbalance is not None:

					# Store the system imbalance
					self.__system_imbalance.update({str(dt_current): pd.read_json(system_imbalance)})
					# logging.info("Current system imbalance {}".format(self.__sys_imbalance[str(dt_current)]))

			# Wait for a moment
			await asyncio.sleep(CF.DELAY)

			# Increment the clock to next period (dt=15min)
			self.container.clock.set_time(self.container.clock.time() + (1*60*CF.GRANULARITY))

			current_datetime = self.container.clock.utcnow().format("YYYY-MM-DD HH:mm:ss")
			# current_datetime = current_datetime + datetime.timedelta(minutes=15)

		
		return True



	def __getLoadPrediction(self, 
		starting_datetime, 
		prediction_window=30):
		"""
		Get the load prediction for next `prediction_window`
		starting from `starting_datetime`
		"""
		# Get the prediction horizon in minute
		prediction_horizon = prediction_window * CF.GRANULARITY


		# Currently, just return the actual load with some noise
		slice_actual = self.__data[str(starting_datetime): 
						str(starting_datetime+datetime.timedelta(minutes=prediction_horizon))][DB.TBL_AGENTS_COLS[1]].copy()
		prediction = np.array(slice_actual)

		# Make a fake prediction
		# probably will be another service that will provide the prediction
		# or implement prediction methodology here somewhere!

		# Fixing the seed
		np.random.seed(100)

		# Create Normally Distributed Random noises (with 10% as standard error)
		rand_noise = np.random.normal(0, 10, len(prediction))/100

		# Add this noise to the actual demand to create a fake predition 
		# profile
		prediction = prediction*(1+rand_noise)

		prediction_df = pd.DataFrame(data=prediction, columns=['load_prediction'])
		prediction_df.index = slice_actual.index

		return prediction_df

