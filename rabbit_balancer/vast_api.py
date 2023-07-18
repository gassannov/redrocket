import time
import subprocess
from loguru import logger
from vast import search__offers

def get_instance(query:list[str]) -> list[str]:
    result = subprocess.run(query, stdout=subprocess.PIPE)
    result = str(result.stdout)
    result = result.replace('\\n', '\\n ')
    enable_instances = result.split('\\n')
    first_instance = enable_instances[1].split(' ')
    enable_instances = [elem for elem in first_instance if len(elem) > 0]
    return first_instance


def get_instances(query:list[str]) -> list[str]:
    result = subprocess.run(query, stdout=subprocess.PIPE)
    result = str(result.stdout)
    result = result.replace('\\n', '\\n ')
    enable_instances = result.split('\\n')
    enable_instances = list(map(lambda x: x.split(' '), enable_instances))
    enable_instances = [list(filter(lambda x: len(x) > 0, instance))
                        for instance in enable_instances]

    return enable_instances


def create_instance():
    logger.debug('start create instance')
    instances = get_instances(['./vast', 'search offers', 'total_flops > 55 disk_space > 15 verified=true gpu_ram > 10'])
    instances = list(filter(lambda x: len(x) > 7, instances))
    instances_sorted = list(sorted(instances, key=lambda x: x[8]))
    instance_id = instances_sorted[20][0]
    #change for inpainting
    print(instance_id)
    result = subprocess.run(f"./vast create instance {instance_id} --image gassanov/inference_server --disk 30 --env '-e port=5672 -e queue_name=inference_queue -e ip=185.145.129.126' --onstart-cmd 'python3 -u receive.py --ip $ip --port_rabbit $port --queue_name $queue_name --skip-torch-cuda-test'", shell=True, stdout=subprocess.PIPE)
    result = str(result.stdout)
    print(result)
    result = int(result[result.rindex(':')+2:result.rindex('}')])
    logger.debug(f'end create instance id = {result}')
    return result


def create_instance_inpainting():
    logger.debug('start create instance')
    instances = get_instances(['./vast', 'search offers', 'total_flops > 55 disk_space > 15 verified=true gpu_ram > 10'])
    instances = list(filter(lambda x: len(x) > 7, instances))
    instances_sorted = list(sorted(instances, key=lambda x: x[8]))
    instance_id = instances_sorted[20][0]
    #change for inpainting
    print(instance_id)
    result = subprocess.run(f"./vast create instance {instance_id} --image gassanov/inpainting_server --disk 30 --env '-e port=5672 -e queue_name=inpainting_queue -e ip=185.145.129.126' --onstart-cmd 'python3 -u receive_inpainting.py --ip $ip --port_rabbit $port --queue_name $queue_name --skip-torch-cuda-test'", shell=True, stdout=subprocess.PIPE)
    result = str(result.stdout)
    print(result)
    result = int(result[result.rindex(':')+2:result.rindex('}')])
    logger.debug(f'end create instance id = {result}')
    return result


def delete_instance(contract_id: int):
    logger.debug(f'delete_instance with contract {contract_id}')
    subprocess.run(f'./vast destroy instance {contract_id}', shell=True)


if __name__ == '__main__':
    create_instance()
    create_instance_inpainting()
