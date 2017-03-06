"""
Blockchain Agent
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
class BlockchainAgent(aiomas.Agent):
	"""
	The residential agents
	"""
	def __init__(self, container, db_engine=None):
		super().__init__(container)

		# Assigning the databse connectin
		# the idea is to create only a single connection
		# Let the mysql's internal connection manager handle
		# all the connection related issue
		self.__conn = db_engine

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

		print(self.__actualData[agent_id])

		return True

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

		print(self.__predictedData[agent_id])

		return True
