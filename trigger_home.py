"""
This module triggers a home agent
provided by the home ID.

The home agent must be located into a container
and ready to be kicked in!

"""
import aiomas
import sys, traceback

class TriggerAgent(aiomas.Agent):
	def __init__(self, container):
		super().__init__(container)

	async def run(self, agent_addr):
		home_agent = await self.container.connect(agent_addr)
		print(home_agent)
		ret = await home_agent.trigger(agent_type="TRIGGER_AGENT")
		print(ret)

# Possibly a database retrieve
agent_lookup = {9019: 'tcp://localhost:5556/0',
				7881: 'tcp://localhost:5556/1',}

agent_id = sys.argv[1]

agent_addr = agent_lookup[int(agent_id)]
print(agent_addr)

c = aiomas.Container.create(('localhost', 5560))

trigger_agent = TriggerAgent(container=c)
aiomas.run(until=trigger_agent.run(agent_addr))