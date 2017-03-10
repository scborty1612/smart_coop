"""
Blockchain observer agent.
This agent basically tries to catch
current status of the block chain agent
and creates or publish some views/plots.
"""

# Import the agent package
import aiomas

# For storing and manipulating data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("../")
from configure import Configure as CF


# Logging stuffs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys, traceback
# Define the generic agent
"""
Later, heterogenious agent will be added 
"""
class BlockchainObserver(aiomas.Agent):
	"""
	The residential agents
	"""
	def __init__(self, container, ):
		super().__init__(container)

		# Store system imbalance
		self.__system_imbalance = None


	async def observeSystemImbalance(self, bc_addr):
		bc_agent = await self.container.connect(bc_addr)

		if not bc_agent:
			return False

		result = await bc_agent.provideSystemImbalance(start_datetime=CF.SIM_START_DATETIME, end_datetime=CF.SIM_END_DATETIME)

		if not result:
			return False

		self.__system_imbalance = pd.read_json(result)

		self.__plotSystemImbalance()

	def __plotSystemImbalance(self):
		"""
		Plotting system imbalance
		"""
		self.__system_imbalance.plot(figsize=(14, 6))
		plt.show()

