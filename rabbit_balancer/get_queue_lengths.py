import requests
from requests.auth import HTTPBasicAuth

QUEUES_ENDPOINT = 'http://185.145.129.126:15672/api/queues/'
RABBIT_LOGIN = 'guest'
RABBIT_PASSWORD = 'guest'


def get_queue_length(queue_name: str) -> int:
    try:
        responce = requests.get(QUEUES_ENDPOINT,
                                auth=HTTPBasicAuth(RABBIT_LOGIN,
                                                   RABBIT_PASSWORD))
    except Exception:
        return 0

    responce_dict = responce.json()
    for queue in responce_dict:
        if queue['name'] == queue_name:
            return queue['messages']

    return 0
