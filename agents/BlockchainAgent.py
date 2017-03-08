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

		logging.info("__________________________")
		logging.info("Agent ID: {}".format(agent_id))
		logging.info(self.__actualData[agent_id])

		return True

	@aiomas.expose
	def provideSystemImbalance(self, start_datetime, end_datetime):
		"""
		Provide the total imbalance (periodically)
		"""
		agents = self.__actualData.keys()

		for i, agent in enumerate(agents):

			cdf = self.__actualData[agent]
			cdf = cdf[str(start_datetime): str(end_datetime)]

			if len(cdf) < 2:
				continue
			cdf['imbalance'] = cdf['grid'] + cdf['gen']

			if i == 0:
				total_imbalance = pd.DataFrame(cdf['imbalance'], columns=['imbalance'])
				total_imbalance.index = cdf.index
			else:
				this_imbalance = np.array(cdf['imbalance'])
				current_imbalance = np.array(total_imbalance['imbalance'])

				if len(this_imbalance) > len(current_imbalance):
					current_imbalance.resize(this_imbalance.shape)
					rs = current_imbalance+this_imbalance
					_index = cdf.index
				else:
					this_imbalance.resize(current_imbalance.shape)
					rs = current_imbalance+this_imbalance
					_index = total_imbalance.index

				total_imbalance = pd.DataFrame(rs, columns=['imbalance'])
				total_imbalance.index = _index


		# Convert it back to dataframe for convenience of
		# jsonify

		return total_imbalance.to_json()

	@aiomas.expose
	def updatePredictedData(self, agent_id, data_serialized):
		# Deserialization 
		predicted_df = pd.read_json(data_serialized)

		if self.__predictedData.get(agent_id) is not None:
			# Get the existing dataframe
			current_pdf = self.__predictedData[agent_id].copy()

			# Combine
			self.__predictedData[agent_id] = predicted_df.combine_first(current_pdf)
		else:
			self.__predictedData[agent_id] = predicted_df

		logging.info(self.__predictedData[agent_id])

		return True
