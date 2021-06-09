import pika
import sys

SPLIT_TOKEN = '||'

client_name = sys.argv[1]
frontend_queue_name = 'frontend'

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost', heartbeat=0)
)
channel = connection.channel()
channel.queue_declare(queue=client_name)
channel.queue_declare(queue=frontend_queue_name)

print(f'>> CLIENT {client_name} STARTED')


def consume_callback(channel, method, header, body):
    print(f'>> RESULT IN {body.decode()}')
    channel.stop_consuming()

try:
    while True:
        # map_file = input('Enter map file name: ')
        groups_num = 'NULL'
        while not groups_num.isdigit():
            groups_num = input('Enter groups number: ')
        result_dir = input('Enter result directory: ')

        body = 'CLI' + SPLIT_TOKEN.join([client_name, groups_num, result_dir])
        channel.basic_publish(exchange='', routing_key=frontend_queue_name, body=body)

        channel.basic_consume(
            on_message_callback=consume_callback,
            queue=client_name,
            auto_ack=True
        )
        channel.start_consuming()
finally:
    channel.queue_purge(queue=client_name)
    connection.close()
    print(f'CLIENT {client_name} STOPPED')
