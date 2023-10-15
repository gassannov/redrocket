import runpod
import time
import subprocess
from loguru import logger

# runpod.api_key = "QA71P9AWWIPTBS9WGR4BU1FO8NLAAQ1EZ6GXJQP1"


def create_instance():
    logger.info('get request for instance creating')
    result = subprocess.run(f"runpodctl create pod --imageName gassanov/inference_server --gpuType 'NVIDIA GeForce RTX 4090' --args 'python3 -u receive.py --skip-torch-cuda-test'", shell=True, stdout=subprocess.PIPE)
    result = str(result.stdout)
    result = result[result.index('"')+1:result.rindex('"')]
    logger.info(f'create instance with id: {result}')
    return result


def create_instance_inpainting():
    logger.info('get request for instance creating')
    result = subprocess.run(f"runpodctl create pod --imageName gassanov/inpainting_server --gpuType 'NVIDIA GeForce RTX 3090' --args 'python3 -u receive_inpainting.py --skip-torch-cuda-test'", shell=True, stdout=subprocess.PIPE)
    result = str(result.stdout)
    result = result[result.index('"')+1:result.rindex('"')]
    logger.info(f'create instance with id: {result}')
    return result


def delete_instance(contract_id: int):
    logger.info(f'delete_instance with contract {contract_id}')
    subprocess.run(f'runpodctl stop pod {contract_id}', shell=True)


def instance_health(instance_id: int) -> bool:
    return True


if __name__ == '__main__':
    create_instance()
    create_instance_inpainting()


# runpodctl create pod --imageName gassanov/inference_server --gpuType 'NVIDIA GeForce RTX 4090' --args 'python3 -u receive.py --ip 185.145.129.126 --port_rabbit 5672 --queue_name inference_queue --skip-torch-cuda-test'