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

import sys
sys.path.append("../")
from configure import Configure as CF

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
	def __init__(self, container, agent_id, db_engine=None,):
		super().__init__(container)
		self.agent_id = agent_id
		logging.info("Agent {}. Address: {} says hi!!".format(agent_id, self))
		logging.info("Agent {} fetching data".format(agent_id))
		logging.info("")

		# Assigning the databse connectin
		# the idea is to create only a single connection
		# Let the mysql's internal connection manager handle
		# all the connection related issue
		self.__conn = db_engine

		# Load the measurement data
		self.__data = self.__loadData()

		# Empty data structure for storing the predictions
		self.__predictions = dict()

		# System imbalannce
		self.__sys_imbalance = dict()

		# Initial battery SOC
		# self.__init_soc = CF.INIT_SOC

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

	def __loadData(self, start_date="2015-01-01", end_date="2015-01-31"):
		"""
		Load the household data for a particular period
		Argument
		"""

		# Form the where clause based on the date filtering
		whereClause = "{} = {}".format(CF.TBL_AGENTS_AGENT_ID_COL, self.agent_id) 

		if start_date and end_date:
			whereClause += " AND date_format(`{}`, '%%Y-%%m-%%d') >= '{}' "\
						   " AND date_format(`{}`, '%%Y-%%m-%%d') < '{}' ".format(CF.TBL_AGENTS_INDEX, start_date, 
						   														  CF.TBL_AGENTS_INDEX, end_date)
		
		# Form the sql query to fetch residential data
		scaled_cols = ",".join(["`{}`*{} as `{}`".format(col, scale, col) 
								for (col, scale) in zip(CF.TBL_AGENTS_COLS, CF.TBL_AGENTS_COLS_SCALE)])
		
		sql_query = "SELECT `{}`, {} FROM `{}` where {}".format(CF.TBL_AGENTS_INDEX, scaled_cols, CF.TBL_ENERGY_USAGE, whereClause)

		# Fetch the data into a pandas dataframe
		df = pd.read_sql(sql_query, self.__conn, parse_dates=[CF.TBL_AGENTS_INDEX], index_col=[CF.TBL_AGENTS_INDEX])

		# df['int_timestamp'] = df['timestamp'].apply(lambda x:int(x.timestamp()))

		if len(df) <= 2:
			# Apparently, no data is there
			return None

		# The columns containing devices
		# consumption_cols = ['air1', 'air2', 'air3', 'airwindowunit1', 'aquarium1', 'bathroom1', 'bathroom2', 
		# 		'bedroom1', 'bedroom2', 'bedroom3', 'bedroom4', 'bedroom5', 'car1', 'clotheswasher1', 
		# 		'clotheswasher_dryg1', 'diningroom1', 'diningroom2', 'dishwasher1', 'disposal1', 'drye1', 
		# 		'dryg1', 'freezer1', 'furnace1', 'furnace2', 'garage1', 'garage2', 'heater1', 
		# 		'housefan1', 'icemaker1', 'jacuzzi1', 'kitchen1', 'kitchen2', 'kitchenapp1', 'kitchenapp2', 
		# 		'lights_plugs1', 'lights_plugs2', 'lights_plugs3', 'lights_plugs4', 'lights_plugs5', 'lights_plugs6', 
		# 		'livingroom1', 'livingroom2', 'microwave1', 'office1', 'outsidelights_plugs1', 'outsidelights_plugs2', 
		# 		'oven1', 'oven2', 'pool1', 'pool2', 'poollight1', 'poolpump1', 'pump1', 'range1', 'refrigerator1', 
		# 		'refrigerator2', 'security1', 'shed1', 'sprinkler1', 'utilityroom1', 'venthood1', 'waterheater1', 
		# 		'waterheater2', 'winecooler1']

		# For now, reduce the DF size 
		# by removing individual loads
		# for col in consumption_cols:
		# 	del df[col]

		# The PV columns may contain negative values 
		# For now, update those values with zero
		
		# 
		gen_col = CF.TBL_AGENTS_COLS[1]
		df[gen_col] = df[gen_col].apply(lambda x: max(x, 0))
	
		logging.info("{}. Total number of records: {}".format(self.agent_id, len(df)))

		return df

	def __scheduleTasks(self):
		"""
		Unused!
		"""
		this_clock = self.container.clock
		gran_sec = CF.granularity * 60

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
			cdf = self.__data[str(CF.SIM_START_DATETIME): str(current_datetime)]

			if cdf is None:
				logger.info("No data is available for the timeframe [{} to {}]".format(str(CF.SIM_START_DATETIME), str(current_datetime)))
				return None

			result = await bc_agent.updateActualData(agent_id=self.agent_id, data_serialized=cdf.to_json())
			# logging.info("received {} from BC agent".format(ret))
		elif mode is 'UPDATE_PREDICTION':
			result = await bc_agent.updatePredictedData(agent_id=self.agent_id, data_serialized=self.__loadPrediction.to_json())

		elif mode is 'RETRIEVE_IMBALANCE':
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
				logging.info("{}. Time for fetching system imbalance from BC...".format(self.agent_id))

				# Communicating with BC
				sys_imbalance = await self.__communicateBlockchain( 
					start_datetime=CF.SIM_START_DATETIME, end_datetime=str(dt_current), 
					mode='RETRIEVE_IMBALANCE')

				# Handling the returned dataframe of overall imbalance
				if sys_imbalance is not None:

					# Store the system imbalance
					self.__sys_imbalance.update({str(dt_current): pd.read_json(sys_imbalance)})
					logging.info("Current system imbalance {}".format(self.__sys_imbalance[str(dt_current)]))

			# Wait for a moment
			await asyncio.sleep(0.5)

			# Increment the clock to next period (dt=15min)
			self.container.clock.set_time(self.container.clock.time() + (1*60*CF.granularity))

			current_datetime = self.container.clock.utcnow().format("YYYY-MM-DD HH:mm:ss")
			# current_datetime = current_datetime + datetime.timedelta(minutes=15)

		
		return True



	def __getLoadPrediction(self, starting_datetime, 
		prediction_window=4):
		"""
		Get the load prediction for next `prediction_window`
		starting from `starting_datetime`
		"""
		# Get the prediction horizon in minute
		prediction_horizon = prediction_window * CF.granularity


		# Currently, just return the actual load with some noise
		slice_actual = self.__data[str(starting_datetime): str(starting_datetime+datetime.timedelta(minutes=prediction_horizon))]['grid'].copy()
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

		# Plot them to have a look

		logging.info("Predcited DF")
		logging.info(prediction_df)

		return prediction_df

