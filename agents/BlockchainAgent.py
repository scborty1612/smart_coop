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
	The residential agents
	"""
	def __init__(self, container, ):
		super().__init__(container)

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

		# logging.info("_________ACTUAL_________________")
		# logging.info("Agent ID: {}".format(agent_id))
		# logging.info(self.__actualData[agent_id])

		return True

	@aiomas.expose
	def provideStaticSystemStatus(self, start_datetime, end_datetime):
		if self.__grid_transfer is None:
			return None
		
		"""
		Prepare the data to be sent over network.
		The method should provide the current status of the system.

		"""

		grid_transfer = None
		if self.__grid_transfer is not None:
			this_grid_transfer = self.__grid_transfer[str(start_datetime): str(end_datetime)]
			grid_transfer = this_grid_transfer.to_json()

		imbalance = None
		if self.__total_imbalance is not None:
			this_imbalance = self.__total_imbalance[str(start_datetime): str(end_datetime)]
			imbalance = this_imbalance.to_json()

		agent_list = [k for k in self.__actualData.keys()]
		
		actual_data = [df[str(start_datetime): str(end_datetime)].to_json() for df in self.__actualData.values()]
		pred_data = [df[str(start_datetime): str(end_datetime)].to_json() for df in self.__predictedData.values()]

		return grid_transfer, imbalance, agent_list, actual_data, pred_data


	@aiomas.expose
	def provideGridifyEnergy(self, start_datetime, end_datetime):
		"""
		Provide the total aggregated energy from/to grid
		"""
		agents = self.__actualData.keys()

		for i, agent in enumerate(agents):

			cdf = self.__actualData[agent]
			cdf = cdf[str(start_datetime): str(end_datetime)]

			if len(cdf) < 2:
				continue
			# Calculate the current imbalance
			cdf['aggregted_grid'] = cdf[DB.TBL_AGENTS_COLS[1]] - cdf[DB.TBL_AGENTS_COLS[0]]

			# Incorporating asynchornously arrival of agent's data
			# into the system imbalance

			if i == 0:
				grid_transfer = pd.DataFrame(cdf['aggregted_grid'], columns=['aggregted_grid'])
				grid_transfer.index = cdf.index
			else:
				this_grid_exchange = np.array(cdf['aggregted_grid'])
				current_imbalance = np.array(grid_transfer['aggregted_grid'])

				# Adjusting the length of the dataframe
				if len(this_grid_exchange) > len(current_imbalance):
					current_imbalance.resize(this_grid_exchange.shape)
					rs = current_imbalance+this_grid_exchange
					_index = cdf.index
				else:
					this_grid_exchange.resize(current_imbalance.shape)
					rs = current_imbalance+this_grid_exchange
					_index = grid_transfer.index

				grid_transfer = pd.DataFrame(rs, columns=['aggregted_grid'])
				grid_transfer.index = _index

		# Store the total imbalance for static transfer
		self.__grid_transfer = grid_transfer

		# Convert it back to dataframe for convenience of
		# jsonify
		return grid_transfer.to_json()


	@aiomas.expose
	def provideSystemImbalance(self, start_datetime, end_datetime):
		"""
		Provide the total aggregated energy from/to grid
		"""
		agents = self.__actualData.keys()

		total_imbalance = None

		for i, agent in enumerate(agents):

			# Take the actual usage data
			if not agent in self.__actualData:
				continue	
			
			actual_df = self.__actualData[agent]

			# Take the predicted usage data
			if not agent in self.__predictedData:
				continue

			predict_df = self.__predictedData[agent]
			
			# print("Actual DF")
			# print(actual_df)
			# print("Predicted DF")
			# print(predict_df)

			# Intersect the data, since the actual data may have additional
			# timestamps
			agg_data = pd.concat([actual_df, predict_df], axis=1, join='inner')

			# cdf = cdf[str(start_datetime): str(end_datetime)]

			if len(agg_data) < 2:
				continue
			
			# Calculate the imbalance (actual_usage - prediction)
			agg_data['imbalance'] = agg_data[DB.TBL_AGENTS_COLS[1]] - agg_data['load_prediction']

			print(agg_data['imbalance'])

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

		# Store the total imbalance for static transfer
		self.__total_imbalance = total_imbalance

		# Convert it back to dataframe for convenience of
		# jsonify
		return total_imbalance.to_json()

	@aiomas.expose
	def updatePredictedData(self, agent_id, data_serialized):
		# Deserialization 
		predicted_df = pd.read_json(data_serialized)

		print("Current Prediction")
		print(predicted_df)

		if self.__predictedData.get(agent_id) is not None:
			# Get the existing dataframe
			current_pdf = self.__predictedData[agent_id].copy()
			print("Existing prediction")
			print(current_pdf)

			# Combine
			self.__predictedData[agent_id] = predicted_df.combine_first(current_pdf)
		else:
			self.__predictedData[agent_id] = predicted_df

		logging.info(self.__predictedData[agent_id])

		return True
