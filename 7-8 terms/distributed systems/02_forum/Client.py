import pika
import sys
import json
import os


class Client:
    def __init__(self, name):
        self.log_time = 0
        self.name = name
        self.state = None

    def print_forum_state(self, id, level, parent_id):
        if self.state is None:
            return
        if id != '0':
            t_log, client_name, message = tuple(self.state[id][:3])
            print(level * '\t' + f'#{id} {client_name} в ответ на #{parent_id} t_log = {t_log}:')
            print(level * '\t' + f'>> {message}')
        for child_id in sorted(self.state[id][3], key=lambda x: self.state[str(x)][0]):
            self.print_forum_state(str(child_id), level + 1, id)

    def consume_callback(self, channel, method, header, body):
        splited = body.decode().split('\t')
        t_log = int(splited[0])
        if self.log_time > t_log:
            self.log_time += 1
        else:
            self.log_time = t_log + 1
        self.state = json.loads(splited[1])
        os.system('clear')
        self.print_forum_state('0', -1, None)
        channel.stop_consuming()

    def start(self, server_name):
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', heartbeat=0))
        channel = connection.channel()
        channel.queue_declare(queue=self.name)
        channel.queue_declare(queue=server_name)
        print(f'>> Client {self.name} started.')
        try:
            while True:
                reply = int(input('На какое сообщение хотите ответить (0 - корневое сообщение): '))
                message = input('Введите сообщение: ')
                body = f'{self.log_time}\t{self.name}\t{message}\t{reply}'
                channel.basic_publish(exchange='', routing_key='server', body=body)
                channel.basic_consume(
                    on_message_callback=self.consume_callback,
                    queue=self.name,
                    auto_ack=True
                )
                channel.start_consuming()
        finally:
            channel.queue_purge(queue=self.name)
            connection.close()
            print(f'>> Client {self.name} closed.')


if __name__ == '__main__':
    client = Client(sys.argv[1])
    client.start(sys.argv[2])