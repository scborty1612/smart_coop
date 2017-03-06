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

import matplotlib.pyplot as plt
import seaborn as sns

import sys, traceback
# Define the generic agent
"""
Later, heterogenious agent will be added 
"""
class PredictionAgent(aiomas.Agent):
	"""
	The residential agents
	"""
	def __init__(self, container, agent_id, db_engine=None):
		super().__init__(container)
		self.agent_id = agent_id

		# Assigning the databse connectin
		# the idea is to create only a single connection
		# Let the mysql's internal connection manager handle
		# all the connection related issue
		self.__conn = db_engine

		# Datastructure for storing the localalized historical data
		# collected from different Home Agents
		self.__historicalData = dict()


	async def collectHistoricalData(self, agent):
		"""
		Pretending as a prediction agent send request to provide
		historical data
		"""

		# Get the historical d
		homeAgent = await self.container.connect(agent.addr)
		historical_data = await homeAgent.provideHistoricalData(sender_agent="PREDICTION_AGENT")

		if historical_data is None:
			print("Nothing is received from {}".format(agent.agent_id))
			return

		# Transform the historical data (in JSON) back to pandas dataframe
		historical_df = pd.read_json(historical_data)

		print("Historical data successfully received from {}".format(agent.agent_id))

		# Keep localize the historical data
		# Only keep the current historical data (overridding the existing data)

		if self.__historicalData.get(agent.agent_id):
			self.__historicalData.udpate({agent.agent_id: historical_df})
		else:
			self.__historicalData[agent.agent_id] = historical_df


	def historicalData(self, agent):
		"""
		Method for testing the validity of the historical data,
		will be definitely removed later
		"""
		return self.__historicalData[agent.agent_id]


	@aiomas.expose
	def getLoadPrediction(self, agent_id, experiment_datetime=None):
		"""
		The prediction service
		"""
		# Retrieve the historical data
		try:
			historical_df = self.__historicalData[agent_id][experiment_datetime]
		except Exception as e:
			print("Probably house {} has no data for {}".format(agent_id, experiment_datetime))
			traceback.print_exc(file=sys.stdout)
			return None
		# print(historical_df[experiment_datetime]['total_energy'])

		# Make a fake prediction
		# probably will be another service that will provide the prediction
		# or implement prediction methodology here somewhere!

		# Fixing the seed
		np.random.seed(100)

		# Create Normally Distributed Random noises (with 10% as standard error)
		rand_noise = np.random.normal(0, 10, len(historical_df))/100

		# Add this noise to the actual demand to create a fake predition 
		# profile
		historical_df['load_prediction'] = historical_df['total_energy']*(1+rand_noise)

		# Plot them to have a look
		historical_df[['total_energy', 'load_prediction']].plot()
		plt.title("Predictions for Home {}".format(agent_id))
		plt.show()
		plt.close()

		# Return the predictions
		# return historical_df.to_json()
		return 93


"""
Create prediction agent
"""

