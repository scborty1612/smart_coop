import aiomas
import asyncio
import time

CLOCK = aiomas.ExternalClock("2015-01-15T00:00:00")

class Sleeper(aiomas.Agent):
	async def run(self):
		print("Gonna sleep for 1s")
		await self.container.clock.sleep(1)

async def clock_setter(factor=0.5):
	while True:
		await asyncio.sleep(factor)
		CLOCK.set_time(CLOCK.time() + 1*60*15)

container = aiomas.Container.create(('localhost', 5555), clock=CLOCK)
t_clock_setter = asyncio.async(clock_setter())

agent = Sleeper(container)
start = time.monotonic()
print(CLOCK.utcnow())
aiomas.run(agent.run())
print(CLOCK.utcnow())
print("Agent process finished after %.1fs"%(time.monotonic()-start))
_ = t_clock_setter.cancel()
container.shutdown()