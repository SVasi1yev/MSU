import pika
import sys
import json
import os


class Server:
    def __init__(self, name):
        self.STATE_FILE = 'forum_state.json'
        self.name = name
        if os.path.isfile(self.STATE_FILE):
            with open(self.STATE_FILE) as f:
                self.log_time = int(f.readline()[:-1])
                self.next_id = int(f.readline()[:-1])
                self.state = json.loads(f.read())
        else:
            self.log_time = 0
            self.next_id = 1
            self.state = {'0': [None, None, None, []]}

    def consume_callback(self, channel, method, header, body):
        splited = body.decode().split('\t')
        t_log = int(splited[0])
        if self.log_time > t_log:
            self.log_time += 1
        else:
            self.log_time = t_log + 1
        client_name = splited[1]
        message = splited[2]
        reply = int(splited[3])
        print(f'>> Get message from {client_name}: {message}')
        if str(reply) in self.state:
            self.state[str(self.next_id)] = [self.log_time, client_name, message, []]
            self.state[str(reply)][3].append(self.next_id)
            self.next_id += 1
        body = str(self.log_time) + '\t' + json.dumps(self.state)
        channel.basic_publish(exchange='', routing_key=client_name, body=body)

    def start(self):
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', heartbeat=0))
        channel = connection.channel()
        channel.queue_declare(queue=self.name)
        channel.basic_consume(
            on_message_callback=self.consume_callback,
            queue=self.name,
            auto_ack=True
        )
        print(f'>> Server {self.name} started.')
        try:
            channel.start_consuming()
        finally:
            channel.queue_purge(queue=self.name)
            connection.close()
            with open(self.STATE_FILE, 'w') as f:
                f.write(str(self.log_time) + '\n')
                f.write(str(self.next_id) + '\n')
                f.write(json.dumps(self.state))
            print(f'>> Server {self.name} closed.')


if __name__ == '__main__':
    server = Server(sys.argv[1])
    server.start()