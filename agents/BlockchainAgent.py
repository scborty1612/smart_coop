"""
Blockchain Agent
"""

# Import the agent package
import aiomas

# For storing and manipulating data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Adding the configuration script
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
class BlockchainAgent(aiomas.Agent):
	"""
	The Blockchain agents
	"""
	def __init__(self, container, SPAgentAddr):
		super().__init__(container)

		# Record SPAgent's address
		self.__spa_addr = SPAgentAddr

		# Local dictionary datastructure
		# to contain the actual demand/generation
		self.__actualData = dict()

		# to store the forecasted/prediction
		# data
		self.__predictedData = dict()

		# Address of the blockchainobserver agent
		self.__bco_address = None

		# static imbalance
		self.__total_imbalance = None

		# Grid exchange
		self.__grid_transfer = None

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
					  agent_id=-2,
					  agent_address=self.addr,
					  agent_type='blockchain',
					  agent_functionality='Blockchain')
		if not status:
			logging.info("Could not register Blockchain agent.")


	def setBCOAddress(self, addr = None):
		"""
		Set the Blockchain Observer agent's address
		"""
		self.__bco_address = addr


	@aiomas.expose
	def updateActualData(self, agent_id, data_serialized):
		"""
		This method updates the actual/realized data
		for a particular time frame.
		"""
		# This part should validate the agent's identification
		# One method to do so, get the agent id by the agent address
		# and match the sender's id


		# Deserialize the data
		actual_df = pd.read_json(data_serialized)

		# First try to retreive the localized actual
		# data for this particular agent

		"""Assume that, the dataframe is indexed by the datatime"""

		if self.__actualData.get(agent_id) is not None:
			# Get the existing dataframe
			current_adf = self.__actualData[agent_id].copy()

			# Combine
			self.__actualData[agent_id] = actual_df.combine_first(current_adf)
		else:
			self.__actualData[agent_id] = actual_df

		logging.info("_________ACTUAL_________________")
		logging.info("Agent ID: {}".format(agent_id))
		# logging.info(np.array(self.__actualData[agent_id]))

		return True

	@aiomas.expose
	def provideSystemStatus(self, start_datetime, end_datetime):
		"""
		Prepare the data to be sent over network.
		The method should provide the current status of the system.
	
		"""

		# Record the overall grid transfer
		grid_transfer = None

		# Calculate the updated grid transfer energy
		print("calculate gridify energy")
		# self.provideGridifyEnergy(start_datetime=start_datetime, end_datetime=end_datetime)
		self.__gridExchange(start_datetime, end_datetime)

		if self.__grid_transfer is not None:
			this_grid_transfer = self.__grid_transfer[str(start_datetime): str(end_datetime)]
			# logger.info("Provided grid exchange energy till {}".format(this_grid_transfer.index[-1]))
			grid_transfer = this_grid_transfer.to_json()

		# Record the overall imbalance
		imbalance = None

		# Calculate the updated system imbalance
		# self.provideSystemImbalance(start_datetime=start_datetime, end_datetime=end_datetime)
		self.__systemImbalance(start_datetime, end_datetime)

		if self.__total_imbalance is not None:
			this_imbalance = self.__total_imbalance[str(start_datetime): str(end_datetime)]
			# logger.info("Provided system imbalance till {}".format(this_imbalance.index[-1]))
			imbalance = this_imbalance.to_json()

		# Lists the agents (with home ID)
		agent_list = [k for k in self.__actualData.keys()]
		
		actual_data = [df[str(start_datetime): str(end_datetime)].to_json() for df in self.__actualData.values()]
		pred_data = [df[str(start_datetime): str(end_datetime)].to_json() for df in self.__predictedData.values()]

		# return grid_transfer, agent_list, actual_data, pred_data
		return grid_transfer, imbalance, agent_list, actual_data, pred_data


	def __gridExchange(self, start_datetime, end_datetime):
		agents = self.__actualData.keys()
		
		grid_transfer = None
		print(agents)
		for i, agent in enumerate(agents):

			cdf = self.__actualData[agent]
			cdf = cdf[str(start_datetime): str(end_datetime)]

			if len(cdf) < 2:
				continue
			# Calculate the current imbalance
			cdf['grid_exchange_wout_battery'] = cdf[DB.TBL_AGENTS_COLS[1]] - cdf[DB.TBL_AGENTS_COLS[0]]

			# Incorporating asynchornously arrival of agent's data
			# into the system imbalance

			if i == 0:
				grid_transfer = pd.DataFrame(cdf['grid_exchange_wout_battery'], 
											columns=['grid_exchange_wout_battery'])
				grid_transfer.index = cdf.index
			else:
				this_grid_exchange = np.array(cdf['grid_exchange_wout_battery'])
				current_imbalance = np.array(grid_transfer['grid_exchange_wout_battery'])

				# Adjusting the length of the dataframe
				if len(this_grid_exchange) > len(current_imbalance):
					current_imbalance.resize(this_grid_exchange.shape)
					rs = current_imbalance+this_grid_exchange
					_index = cdf.index
				else:
					this_grid_exchange.resize(current_imbalance.shape)
					rs = current_imbalance+this_grid_exchange
					_index = grid_transfer.index

				grid_transfer = pd.DataFrame(rs, columns=['grid_exchange_wout_battery'])
				grid_transfer.index = _index

		if grid_transfer is None:
			return None

		# Store the total imbalance for static transfer
		self.__grid_transfer = grid_transfer

		# Convert it back to dataframe for convenience of
		# jsonify
		return grid_transfer.to_json()


	@aiomas.expose
	def provideGridifyEnergy(self, start_datetime, end_datetime):
		"""
		Provide the total aggregated energy from/to grid
		"""
		# print("calculate gridify energydddddddd")
		return self.__gridExchange(start_datetime, end_datetime)


	def __systemImbalance(self, start_datetime, end_datetime):

		agents = self.__actualData.keys()
		logging.info("System imbalance 1")

		total_imbalance = None

		for i, agent in enumerate(agents):
			# If the agent doesnt yet contain the actual demand data
			if not agent in self.__actualData:
				continue	

			# Take the actual usage data			
			actual_df = self.__actualData[agent]

			# Take the predicted usage data
			if not agent in self.__predictedData:
				continue

			predict_df = self.__predictedData[agent]

			# Intersect the data, since the actual/predicted data may have additional
			# timestamps
			# predict_df.to_csv("pdf_{}.csv".format(agent))
			# actual_df.to_csv("adf_{}.csv".format(agent))
			agg_data = pd.concat([actual_df, predict_df], axis=1, join='inner')

			# cdf = cdf[str(start_datetime): str(end_datetime)]

			if len(agg_data) < 2:
				continue
			
			# Calculate the imbalance (actual_usage - prediction)
			agg_data['imbalance'] = agg_data[DB.TBL_AGENTS_COLS[1]] - agg_data['load_prediction']
			# agg_data.to_csv("agg_data_{}.csv".format(agent))

			# print(agg_data['imbalance'])

			# Incorporating asynchornously arrival of agent's data
			# into the system imbalance

			if i == 0:
				total_imbalance = pd.DataFrame(agg_data['imbalance'], columns=['imbalance'])
				total_imbalance.index = agg_data.index
			else:
				this_imbalance = np.array(agg_data['imbalance'])
				current_imbalance = np.array(total_imbalance['imbalance'])

				# Adjusting the length of the dataframe
				if len(this_imbalance) > len(current_imbalance):
					current_imbalance.resize(this_imbalance.shape)
					rs = current_imbalance+this_imbalance
					_index = agg_data.index
				else:
					this_imbalance.resize(current_imbalance.shape)
					rs = current_imbalance+this_imbalance
					_index = total_imbalance.index

				total_imbalance = pd.DataFrame(rs, columns=['imbalance'])
				total_imbalance.index = _index

		# return nothing in case the total imbalanc is not initiated
		if total_imbalance is None:
			return None
		# total_imbalance.to_csv("total_imbalanc.csv")
		# logging.info("System imbalance")
		# logging.info(total_imbalance)

		# Store the total imbalance for static transfer
		self.__total_imbalance = total_imbalance

		# Convert it back to dataframe for convenience of
		# jsonify
		return total_imbalance.to_json()

	@aiomas.expose
	def provideSystemImbalance(self, start_datetime, end_datetime):
		"""
		Provide the total system imbalace realize so far.
		The difference between predicted demand and the realized one
		"""
		return self.__systemImbalance(start_datetime, end_datetime)

	@aiomas.expose
	def updatePredictedData(self, agent_id, data_serialized):
		# Deserialization 
		predicted_df = pd.read_json(data_serialized)

		# print("Current Prediction")
		# print(predicted_df)

		if self.__predictedData.get(agent_id) is not None:
			# Get the existing dataframe
			current_pdf = self.__predictedData[agent_id].copy()
			# print("Existing prediction")
			# print(current_pdf)

			# Combine
			self.__predictedData[agent_id] = predicted_df.combine_first(current_pdf)
		else:
			self.__predictedData[agent_id] = predicted_df

		logging.info("Updating prediction for {}".format(agent_id))
		# logging.info(np.array(self.__predictedData[agent_id]))
		# logging.info(self.__predictedData[agent_id])

		return True
