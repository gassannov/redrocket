from rabbit_listener import Listener
from vast_api import create_instance
import asyncio
from loguru import logger
from get_queue_lengths import get_queue_length
import click


THRESHOLD = 5


#change for inpainting
def create_condition(queue_name='inference_queue') -> bool:
    return get_queue_length(queue_name) > THRESHOLD


#change for inpainting
def breack_condition(queue_name='inference_queue') -> bool:
    return get_queue_length(queue_name) < THRESHOLD


if __name__ == '__main__':
    # print(get_queue_length('inference_queue'))
    logger.add('logs/rabbit_balancer.log')
    listener = Listener(instance_break_condition=breack_condition,
                        instance_create_condition=create_condition,
                        pause_time=1,
                        callback_time=400,
                        instance_create_time=200,
                        instance_pause_time=2)
    asyncio.run(listener.start_listen())
