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
class HomeAgent(aiomas.Agent):
	"""
	The residential agents
	"""
	def __init__(self, container, agent_id, db_engine=None):
		super().__init__(container)
		self.agent_id = agent_id
		print("Agent {}. Address: {} says hi!!".format(agent_id, self))
		print("Agent {} fetching data".format(agent_id))
		print("")

		# Assigning the databse connectin
		# the idea is to create only a single connection
		# Let the mysql's internal connection manager handle
		# all the connection related issue
		self.__conn = db_engine

		self.__data = self.__loadData()

		
	def __loadData(self, start_date="2015-01-01", end_date="2015-01-31"):
		"""
		Load the household data for a particular period
		Argument
		"""

		# Form the where clause based on the date filtering
		whereClause = "house_id = {}".format(self.agent_id) 

		if start_date and end_date:
			whereClause += " AND date_format(`timestamp`, '%%Y-%%m-%%d') >= '{}' "\
						   " AND date_format(`timestamp`, '%%Y-%%m-%%d') < '{}' ".format(start_date, end_date)

		# Form the sql query to fetch residential data
		sql_query = "SELECT * FROM `ps_energy_usage_15min` where {}".format(whereClause)

		# print(sql_query)
		# Fetch the data into a pandas dataframe
		df = pd.read_sql(sql_query, self.__conn, parse_dates=['timestamp'], index_col=['timestamp'])

		# df['int_timestamp'] = df['timestamp'].apply(lambda x:int(x.timestamp()))

		del df['row_id']

		if len(df) <= 2:
			# Apparently, no data is there
			return None

		# The columns containing devices
		consumption_cols = ['air1', 'air2', 'air3', 'airwindowunit1', 'aquarium1', 'bathroom1', 'bathroom2', 
				'bedroom1', 'bedroom2', 'bedroom3', 'bedroom4', 'bedroom5', 'car1', 'clotheswasher1', 
				'clotheswasher_dryg1', 'diningroom1', 'diningroom2', 'dishwasher1', 'disposal1', 'drye1', 
				'dryg1', 'freezer1', 'furnace1', 'furnace2', 'garage1', 'garage2', 'heater1', 
				'housefan1', 'icemaker1', 'jacuzzi1', 'kitchen1', 'kitchen2', 'kitchenapp1', 'kitchenapp2', 
				'lights_plugs1', 'lights_plugs2', 'lights_plugs3', 'lights_plugs4', 'lights_plugs5', 'lights_plugs6', 
				'livingroom1', 'livingroom2', 'microwave1', 'office1', 'outsidelights_plugs1', 'outsidelights_plugs2', 
				'oven1', 'oven2', 'pool1', 'pool2', 'poollight1', 'poolpump1', 'pump1', 'range1', 'refrigerator1', 
				'refrigerator2', 'security1', 'shed1', 'sprinkler1', 'utilityroom1', 'venthood1', 'waterheater1', 
				'waterheater2', 'winecooler1']

		# Other summer data
		pv_gen_col = ['gen']
		grid_col = ['grid']
		act_con_col = ['use']

		df['total_energy'] = np.zeros(len(df))
		df['total_energy'] = df[consumption_cols].sum(axis=1)

		# print(df[cols])

		for col in consumption_cols:
			del df[col]


		return df


	async def run(self, addr):
		"""
		Dropping in here from the test code.
		Will be integrated for inter-agent communications.

		"""
		remote_agent = await self.container.connect(addr)
		ret = await remote_agent.service(42)
		print("{} got {} from {}".format(self.agent_id, ret, remote_agent.agent_id))



	async def getLoadPrediction(self, prediction_agent, experiment_datetime=None):
		"""
		Get the predictions for the experimented date time
		from prediction class
		"""

		# Connect to the prediction agent
		predictionAgent = await self.container.connect(prediction_agent.addr)

		# Collec the predictions
		predictions = await predictionAgent.getLoadPrediction(self.agent_id, experiment_datetime)


	@aiomas.expose
	def provideHistoricalData(self, sender_agent=None):
		"""
		Just make sure, only provide historical data to the 
		prediction.
		"""
		# For now, do it in old and dirty way
		# print("Sender agent is {}".format(sender_agent))
		if sender_agent != 'PREDICTION_AGENT':
			print("Unknown request from fishy agent {}".format(sender_agent))
			return None

		# Check whether any historical data exists
		# for this agent
		if self.__data is None or len(self.__data) <= 2:
			print("Agent {} has no historical data".format(self.agent_id))
			return None

		"""Localize the dataframe"""

		# Just send only what is required
		required_cols = ['total_energy', 'gen']

		# Take a snapshot of the data
		this_data = self.__data[required_cols].copy()
		
		# Jason-i-fy
		ret_obj = this_data.to_json()

		return ret_obj
