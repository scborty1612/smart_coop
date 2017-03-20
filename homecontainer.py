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
import click

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


"""
CLI for triggering home agent
"""

@click.command()
@click.option('--session-id', required=True,
              help="Session ID to root agetns")

def main(session_id):	

	# Creating container that contains the home agents
	HC = aiomas.Container.create(('localhost', 5556), clock=CLOCK)

	# List of Homes
	homes = [9019, 7850, ]# 7881, 100237, 9981, 980,]

	# Creating the session
	spa_addr = DB.getServiceProviderAgent(session_id)

	# Initiate the agents into HC
	homeAgents = [HomeAgent(container=HC, agent_id=home, spa_addr=spa_addr) for home in homes]

	# Register the home agent to the SPA and also bind the associated
	# blockchain agent
	for agent in homeAgents:
		try:
			aiomas.run(until=agent.registerAndBind(session_id=session_id))
		except Exception as e:
			traceback.print_exc(file=sys.stdout)

	# Run the event loop
	try:
		logger.info("Running the event loop. One of the home agents is trying to connect with BC agent!")
		# logger.info("Session ID:{}".format(session_id))

		# Run the even loop 
		aiomas.run()
	except KeyboardInterrupt:
		logger.info("Stopping the event loop")
		# Try to stop the event loop

	except Exception as e:
		traceback.print_exc(file=sys.stdout)
	finally:
		# Killing the current session
		for agent in homeAgents:
			try:
				aiomas.run(until=agent.kill())
			except Exception as e:
				traceback.print_exc(file=sys.stdout)
		

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
	main()
