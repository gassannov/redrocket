from rabbit_listener import Listener
import asyncio
from loguru import logger
from get_queue_lengths import get_queue_length
import argparse
from runpod_api import create_instance, create_instance_inpainting


THRESHOLD = 5
queue_name = 'inference_queue'


def create_condition() -> bool:
    return get_queue_length(queue_name) > THRESHOLD


def breack_condition() -> bool:
    return get_queue_length(queue_name) < THRESHOLD


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='Mode (inpainting or inference, default: inference)', default='inference')
    args = parser.parse_args()

    if args.mode == 'inference':
        create_function = create_instance
        queue_name = 'inference_queue'
        logger.add('logs/rabbit_balancer_inference.log')

    elif args.mode == 'inpainting':
        create_function = create_instance_inpainting
        queue_name = 'inpainting_queue'
        logger.add('logs/rabbit_balancer_inpainting.log')

    else:
        raise Exception('Mode should be inference or inpainting')

    listener = Listener(instance_break_condition=breack_condition,
                        instance_create_condition=create_condition,
                        pause_time=1,
                        callback_time=300,
                        instance_create_time=100,
                        instance_pause_time=2,
                        create_function=create_function)
    asyncio.run(listener.start_listen())
