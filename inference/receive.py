from launch import prepare_environment
prepare_environment()
from modules import shared
from modules.call_queue import wrap_gradio_gpu_call
from modules.txt2img import txt2img
from modules.img2img import img2img
import pickle
from webui import initialize
import requests
import json
import base64
import os
import time
import pika
import sys
import shutil
import io
from PIL import Image
from loguru import logger

NEGATIVE_PROMPT = "((blurry)), animated, cartoon, duplicate, child, childish, paintings, sketches, (worst quality:2), (low quality:2), lowres, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (((pubic hair:2))), blurry, cropped head, 3d, render, extra legs, extra arms, extra fingers, 6 or more fingers per hand, extra torso, extra head, missing leg, missing arm, missing fingers, merged arms, merged legs, merged torso, wrinkle on face, watermarks"


def custom_inference(prompt, seed, batch_size):
    wrapped_txt2img = wrap_gradio_gpu_call(txt2img, extra_outputs=[None, '', ''])
    result = wrapped_txt2img('task(7ldab5vorz522sg)', prompt, NEGATIVE_PROMPT, [], 20, 0, False, False, 1, batch_size, 7, seed, -1.0, 0, 0, 0, False, 512, 512, False, 0.7, 2, 'Latent', 0, 0, 0, [], 0, False, False, 'positive', 'comma', False, False, '', 1, '', 0, '', 0, '', True, False, False, False)
    return result


def img2img_inference(image, prompt, seed, batch_size):
    wrapped_img2img = wrap_gradio_gpu_call(img2img, extra_outputs=[None, '', ''])
    result = wrapped_img2img('', 0, prompt, NEGATIVE_PROMPT, [], image, None, None, None, None, None, None, 20, 0,
                             4, 0, 1, False, False, 1, batch_size, 7, 1.5, 0.75, seed, -1.0, 0, 0, 0, False, 512,
                             512, 0, 0, 32, 0, '', '', '', '',  0, False, 'none', None, 1, None, False, 'Scale to Fit(Inner Fit)',
                             False, False, 64, 64, 64, 1, False)
    return result


def callback(ch, method, properties, body):
    message = 'ok'
    try:
        request_params = json.loads(body)
        logger.info(f'received {str(body)[:50]}')
    except Exception as e:
        message = str(e)
        logger.error(message)
        resp_dict = {'message': message}
        try:
            ch.basic_publish(exchange='',
                            routing_key=properties.reply_to,
                            properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                            body=json.dumps(resp_dict))
        except Exception as e:
            logger.error(f'error to publish: {e}')
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    batch_size = request_params['batch_size'] if 'batch_size' in request_params else 1
    prompt = request_params['prompt'] if 'prompt' in request_params else ''
    seed = request_params['seed'] if 'seed' in request_params else 0
        
    if prompt == '':
        print('wrong prompt')
        message = 'error: prompt must be defined'

    resp_dict = {}

    if 'type' not in request_params:
        logger.error('wrong type')
        message = 'error: type must be defined'

    elif request_params['type'] == 'from_text':
        res = custom_inference(prompt, seed, batch_size)
        
    elif request_params['type'] == 'variations':
        if 'input_image' in request_params:
            base64_code = request_params['input_image']
            img = Image.open(io.BytesIO( base64.b64decode(base64_code) ) )
            res = img2img_inference(img, request_params['prompt'], request_params['seed'], request_params['batch_size']) 
        else:
            logger.error('wrong input_image')
            message = 'error: variations need "input_image" in json query'

    else:
        logger.error('wrong type format')
        message = 'error: wrong format type("from_text" or "variations")'

    resp_dict['message'] = message
    if message != 'ok':
        try:
            ch.basic_publish(exchange='',
                             routing_key=properties.reply_to,
                             properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                             body=json.dumps(resp_dict))
        except Exception as e:
            print('error to publish: ', str(e))
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return
      
    for i, img in enumerate(res[0]):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        resp_dict[f'image{i}'] = img_str.decode('utf-8')

    try:
        ch.basic_publish(exchange='',
                         routing_key=properties.reply_to,
                         properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                         body=json.dumps(resp_dict))
        logger.info(f'publish message from body: {body[:10]}')
    except Exception as e:
        logger.error(f'unexpected error while publish: {e}', )
    ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    port = int(shared.cmd_opts.port)
    ip = shared.cmd_opts.ip
    queue_name = shared.cmd_opts.queue_name
    logger.info(f'start service with [port: {port}, ip: {ip}, queue_name: {queue_name}]')
    initialize()
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=ip, port=port))
    channel = connection.channel()

    channel.queue_declare(queue=queue_name)
    
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=queue_name, on_message_callback=callback)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    try:
        logger.remove()
        logger.add('logs/inference.log', level='DEBUG', rotation='5 MB')
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(1)