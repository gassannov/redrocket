from vast_api import create_instance, delete_instance
import time
import asyncio
from goroutine.app import go
from loguru import logger


async def instance_listen(time_after_create: int,
                          time_pause: int,
                          break_condition):
    is_first = True
    logger.info('instance listen ready')
    contract_id = create_instance()
    logger.info('instance listen start')
    i = 0
    while True:
        if is_first:
            await asyncio.sleep(time_after_create)
        else:
            await asyncio.sleep(time_pause)
        if break_condition() or i == 18000:
            delete_instance(contract_id)
            break
        i += 1
        is_first = False


class Listener():
    def __init__(self, instance_create_condition,
                 instance_break_condition,
                 pause_time=1,
                 callback_time=200,
                 instance_create_time=3600,
                 instance_pause_time=10) -> None:
        self.instance_break_condition = instance_break_condition
        self.instance_create_condition = instance_create_condition
        self.instance_create_time = instance_create_time
        self.instance_pause_time = instance_pause_time
        self.callback_time = callback_time
        self.pause_time = pause_time

    async def start_listen(self):
        logger.info('start listen main listener')
        while True:
            if self.instance_create_condition():
                logger.info('instance listen ready')
                go(instance_listen,
                   self.instance_create_time,
                   self.instance_pause_time,
                   self.instance_break_condition)
                logger.info('instance create after pause')
                time.sleep(self.callback_time)
            else:
                time.sleep(self.pause_time)