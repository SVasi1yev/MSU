import pika
from collections import deque
import time

SPLIT_TOKEN = '||'

frontend_queue_name = 'frontend'
master_queue_name = 'master'

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost', heartbeat=0)
)
channel = connection.channel()
channel.queue_declare(queue=frontend_queue_name)
channel.queue_purge(queue=frontend_queue_name)
channel.queue_declare(queue=master_queue_name)
channel.queue_purge(queue=master_queue_name)
tasks_queue = deque()
timeout = 10
master_timeout = 20
last_job_time = None
cur_task = 0

print(f'>> FRONTEND {frontend_queue_name} STARTED')

try:
    for method_frame, properties, body in channel.consume(frontend_queue_name, inactivity_timeout=timeout):
        if body is not None:
            body = body.decode()
            print(f'>> GET {body}')
            if body.startswith('CLI'):
                tasks_queue.append(str(cur_task) + SPLIT_TOKEN + body[3:])
                cur_task += 1
            elif body.startswith('MAS'):
                tasks_queue.popleft()
                last_job_time = None
                client_name, result_dir = tuple(body[3:].split(SPLIT_TOKEN))
                channel.basic_publish(exchange='', routing_key=client_name, body=result_dir)
            else:
                print('ERROR')
        if (last_job_time is None or (time.time() - last_job_time) > master_timeout) and len(tasks_queue) > 0:
            task = tasks_queue.popleft()
            tasks_queue.appendleft(task)
            last_job_time = time.time()
            channel.basic_publish(exchange='', routing_key=master_queue_name, body=task)
            print(f'>> SEND {task} TO MASTER')
        if body is not None:
            channel.basic_ack(method_frame.delivery_tag)
finally:
    channel.queue_purge(queue=frontend_queue_name)
    connection.close()
    print(f'FRONTEND {frontend_queue_name} STOPPED')