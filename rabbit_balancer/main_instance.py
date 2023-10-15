from runpod_api import create_instance, instance_health
import time
from loguru import logger


if __name__ == '__main__':
    instance_id = create_instance()
    logger.add('logs/rabbit_balancer_inference.log')
    logger.info('create starting main instance')
    while True:
        if not instance_health(instance_id):
            logger.error('main instance fail')
            instance_id = create_instance()
            logger.info('create main instance after fail')
        time.sleep(1)
