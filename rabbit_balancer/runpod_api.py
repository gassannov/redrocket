import runpod

runpod.api_key = "QA71P9AWWIPTBS9WGR4BU1FO8NLAAQ1EZ6GXJQP1"

gpus = runpod.get_gpus()

for gpu in gpus:
    print(gpu)


# runpodctl create pod --imageName gassanov/inference_server --gpuType 'NVIDIA GeForce RTX 4090' --args 'python3 -u receive.py --ip 185.145.129.126 --port_rabbit 5672 --queue_name inference_queue --skip-torch-cuda-test'