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
import math

# from detect_age import detect_age

pixels_limit = 1920*1080

NEGATIVE_PROMPT = "((blurry)), animated, cartoon, duplicate, child, childish, paintings, sketches, (worst quality:2), (low quality:2), lowres, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (((pubic hair:2))), blurry, cropped head, 3d, render, extra legs, extra arms, extra fingers, 6 or more fingers per hand, extra torso, extra head, missing leg, missing arm, missing fingers, merged arms, merged legs, merged torso, wrinkle on face, watermarks"


def add_border(img, border_size: tuple):
    new_size = border_size
    new_image = Image.new("RGB", new_size)
    box = tuple((n - o) // 2 for n, o in zip(new_size, img.size))
    new_image.paste(img, box)
    return new_image, box, img.size


def inpainting_inference(image, mask, prompt, seed, batch_size, height=512, width=512):
    wrapped_img2img = wrap_gradio_gpu_call(img2img,
                                           extra_outputs=[None, '', ''])
#     params = {'id_task': 'task(nk26d8lhhsbnprv)', 'mode': 2, 'prompt': 'a fat yellow cat sit on a bench', 'negative_prompt': '', 'prompt_styles': [], 'init_img': None, 'sketch': None, 'init_img_with_mask': {'image': <PIL.Image.Image image mode=RGBA size=512x512 at 0x7F157D2E9E40>, 'mask': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=512x512 at 0x7F157D2EBBE0>}, 'inpaint_color_sketch': None, 'inpaint_color_sketch_orig': None, 'init_img_inpaint': None, 'init_mask_inpaint': None, 'steps': 20, 'sampler_index': 0, 'mask_blur': 4, 'mask_alpha': 0, 'inpainting_fill': 1, 'restore_faces': False, 'tiling': False, 'n_iter': 1, 'batch_size': 1, 'cfg_scale': 7, 'image_cfg_scale': 1.5, 'denoising_strength': 0.75, 'seed': -1.0, 'subseed': -1.0, 'subseed_strength': 0, 'seed_resize_from_h': 0, 'seed_resize_from_w': 0, 'seed_enable_extras': False, 'selected_scale_tab': 0, 'height': 512, 'width': 512, 'scale_by': 1, 'resize_mode': 0, 'inpaint_full_res': 0, 'inpaint_full_res_padding': 32, 'inpainting_mask_invert': 0, 'img2img_batch_input_dir': '', 'img2img_batch_output_dir': '', 'img2img_batch_inpaint_mask_dir': '', 'override_settings_texts': [], 'args': (0, <scripts.controlnet_ui.controlnet_ui_group.UiControlNetUnit object at 0x7f157d2ebee0>, '<ul>\n<li><code>CFG Scale</code> should be 2 or lower.</li>\n</ul>\n', True, True, '', '', True, 50, True, 1, 0, False, 4, 0.5, 'Linear', 'None', '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>', 128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'], False, False, 'positive', 'comma', 0, False, False, '', '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>', 64, 0, 2, 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0, None, None, False, 50), 'override_settings': {}}

#     result = wrapped_img2img("", 4, prompt, NEGATIVE_PROMPT, [], None, None,
#         None, None, None, image, mask, 20, 0, 4, 0, 1, False, False, 1, 1, 7, 1.5,
#         0.75, seed, -1.0, 0, 0, 0, False, 512, 512, 0, 0, 32, 0, '', '', '', [],
#         0, '<ul>\n<li><code>CFG Scale</code> should be 2 or lower.</li>\n</ul>\n', True, True, '', '', True, 50, True, 1, 0, False, 4, 1, 'None', '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>', 128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'], False, False, 'positive', 'comma', 0, False, False, '', '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>', 64, 0, 2, 1, '', 0, '', 0, '', True, False, False, False, 0
#     )
    result = wrapped_img2img('task(nk26d8lhhsbnprv)', 4, prompt, NEGATIVE_PROMPT, [], None, None, None, None, None, image, mask, 20, 0, 10, 10, 10, False, False, 1, 1, 7, 1.5, 0.75, -1.0, -1.0, 0, 0, 0, False, 0, height, width, 1, 0, 0, 32, 0, '', '','', [], 0, None, '<ul>\n<li><code>CFG Scale</code> should be 2 or lower.</li>\n</ul>\n', True, True, '', '', True, True, '', '', True, 50, True, 1, 0, False, 4, 0.5, 'Linear', 'None', '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>', 128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'], False, False, 'positive', 'comma', 0, False, False, '', '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>', 64, 0, 2, 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0, None, None, False, 50)
    return result


def callback(ch, method, properties, body):
    print('received message')
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
            logger.error(f'unexpected error while publish: {e}', )
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    batch_size = request_params['batch_size'] if 'batch_size' in request_params else 1
    prompt = request_params['prompt'] if 'prompt' in request_params else ''
    seed = request_params['seed'] if 'seed' in request_params else 0

    if prompt == '':
        logger.error('received messge without prompt')
        message = 'error: prompt must be defined'

    resp_dict = {}

    if 'mask' not in request_params:
        logger.error('received messge without mask')
        message = 'error: mask should be in params'

    elif 'input_image' not in request_params:
        logger.error('received message without input_image')
        message = 'error: inpainting need "input_image" in json query'

    else:
        try:
            base64_code = request_params['input_image']
            img = Image.open(io.BytesIO(base64.b64decode(base64_code)))
            # img.save('inpainting_input_image.jpg')
            age = '(20-24)'
#             try:
#                 age = detect_age('inpainting_input_image.jpg')[0]
#             except Exception as e:
#                 age = '(25-32)'
#                 logger.error(f'can not detect age:{e}')
            logger.info(f'detected age: {age}')
            if age != '(0-2)' and age != '(4-6)' and age != '(8-12)':
                base64_mask_code = request_params['mask']
                mask = Image.open(io.BytesIO(base64.b64decode(base64_mask_code)))
                w, h = img.size
                new_w, new_h = w, h
                print('before compressing', w, h)
                num_pixels = w*h
                compress_koeff = 1
                if num_pixels > pixels_limit:
                    compress_koeff = pixels_limit/num_pixels
                    new_h = int(h*math.sqrt(compress_koeff))
                    new_w = int(w*math.sqrt(compress_koeff))
                img = img.resize((new_w, new_h))
                mask = mask.resize((new_w, new_h))
                print('after compressing', img.size)
                res = inpainting_inference(img, mask, prompt, seed, batch_size, new_h, new_w)
                logger.info('succesfully generate image')
            else:
                message = 'error: child deteced'
                logger.info('child deteced')
        except Exception as e:
            message = 'error: making image'
            logger.error(f'unexpected error while making image:{e}')

    resp_dict['message'] = message
    if message != 'ok':
        try:
            ch.basic_publish(exchange='',
                             routing_key=properties.reply_to,
                             properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                             body=json.dumps(resp_dict))
        except Exception as e:
            logger.error('unexpected error while publish: {e}', )
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    for i, img in enumerate(res[0]):
        try:
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            resp_dict[f'image{i}'] = img_str.decode('utf-8')
        except Exception as e:
            resp_dict = {'message': 'error while make base64 from result'}
            logger.error(f'unexpected error while make base64 from result: {e}')

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
#     port = int(shared.cmd_opts.port)
    port = 5672
#     ip = shared.cmd_opts.ip
#     queue_name = shared.cmd_opts.queue_name
    queue_name = 'inpainting_queue'
    ip = '185.145.129.126'
    initialize()
    logger.info(f'start service with [port: {port}, ip: {ip}, queue_name: {queue_name}]')
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=ip, port=port))
    channel = connection.channel()

    channel.queue_declare(queue=queue_name)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=queue_name, on_message_callback=callback)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    try:
        channel.start_consuming()
    except Exception as e:
        logger.error(f'failed rabbit connection: {e}')


if __name__ == '__main__':
    try:
        logger.remove()
        logger.add('logs/inpainting_logs.log', level='DEBUG', rotation='5 MB')
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(1)
