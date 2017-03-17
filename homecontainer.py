"""
This module contains the Home Container that hosts
home agents.

"""
import aiomas
from agents.BlockchainAgent import BlockchainAgent
from agents.HomeAgent import HomeAgent
from configure import Configure as CF
from util import DBGateway as DB

# Database related stuff
from sqlalchemy import create_engine

import pandas as pd

# System and traceback
import sys, traceback

# Logging stuffs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clock related stuff
import asyncio
import time
import datetime
import random

# Agree on clocking
CLOCK = aiomas.ExternalClock(CF.SIM_START_DATETIME)

# Setting up the clock setting function
async def clock_setter(factor=10.):
	while True:
		await asyncio.sleep(factor)

		# delta t is 15min
		CLOCK.set_time(CLOCK.time() + (1*60*15))



def runContainer():	

	# Creating container contains the home agents
	# and prediction agents
	HC = aiomas.Container.create(('localhost', 5556), clock=CLOCK)
	# HC = aiomas.Container.create(('localhost', 5556), clock=CLOCK)

	# Set the clcok
	# t_clock_setter = asyncio.async(clock_setter())


	# List of Homes
	homes = [9019, 7850, ]# 7881, 100237, 9981, 980,]


	# Initiate the agents into HC
	homeAgents = [HomeAgent(container=HC, agent_id=home,) for home in homes]

	# Creating the session
	session_id = DB.createSession(agents=homeAgents)

	# Address of the blockchain agent
	# Later, it will be retreived from the Agent Server
	bc_address = DB.getActiveBlockchainAddress()

	if bc_address is None:
		logging.info("Blockchain is not initiated.")
	else:
		# Bind the blockchain with home agents
		for agent in homeAgents:
			agent.setBlockchainAddress(bc_address=bc_address)

	# Run the event loop
	try:
		logger.info("Running the event loop. One of the home agents is trying to connect with BC agent!")
		logger.info("Session ID:{}".format(session_id))

		# Run the even loop 
		aiomas.run()
	except KeyboardInterrupt:
		logger.info("Stopping the event loop")
		# Try to stop the event loop

	except Exception as e:
		traceback.print_exc(file=sys.stdout)
	finally:
		# Killing the current session
		DB.killSession(session_id=session_id)
		

	# Shutting donw the controller and thereby cleaning 
	# all agents
	try:
		logger.info("Shutting down the home container...and cancelling the clock")
		HC.shutdown()
		# t_clock_setter.cancel()
		logger.info("Done.")
	except Exception as e:
		logger.info("Failed to shutdown the home container")
		traceback.print_exc(file=sys.stdout)


if __name__ == '__main__':
	runContainer()
