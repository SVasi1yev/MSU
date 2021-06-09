import pika
import geopandas as gpd
from shapely.geometry import Polygon
import shutil
import os
import sys
import time
import threading
import multiprocessing

SPLIT_TOKEN = '||'

frontend_queue_name = 'frontend'
master_queue_name = 'master'


class Worker:
    def __init__(self, name, master_name, process_num):
        self.name = name
        self.master_name = master_name
        self.master = name == master_name
        self.us = gpd.read_file('us.geojson')
        self.us = self.us.drop(3).drop(50)
        bounds = self.us.bounds
        self.minx = bounds['minx'].min()
        self.miny = bounds['miny'].min()
        self.maxx = bounds['maxx'].max()
        self.maxy = bounds['maxy'].max()
        self.workers = [f'{i}' for i in range(process_num)]
        self.timeout = 4
        self.service_thread = None
        self.service_running = False
        self.last_task = -1

    def get_service_name(self, name):
        return f'{name}_service'

    def get_grid(self, n):
        for i in range(int(n**0.5), 0, -1):
            if n % i == 0:
                return i, n // i

    def process_job(self, job):
        splited = job.split(SPLIT_TOKEN)
        job_x, job_y = tuple(map(int, splited[0].split(';')))
        result_dir = splited[1]
        points = []
        for coords in splited[2:]:
            points.append(tuple(map(float, coords.split(';'))))
        d = {'geometry': [Polygon(points)]}
        gdf = gpd.GeoDataFrame(d, crs='EPSG:4326')
        res = gpd.overlay(self.us, gdf)
        time.sleep(3)
        with open(f'{result_dir}/{job_x}_{job_y}.geojson', 'w') as f:
            f.write(res.to_json())

    def service(self):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost', heartbeat=0)
        )
        channel = connection.channel()
        channel.queue_declare(queue=self.get_service_name(self.name))
        channel.queue_purge(queue=self.get_service_name(self.name))
        time.sleep(self.timeout)
        try:
            while self.service_running:
                if self.master:
                    for w in self.workers:
                        if w != self.name:
                            channel.basic_publish(exchange='', routing_key=self.get_service_name(w), body='PING')
                    workers = self.workers[:]
                    workers.remove(self.name)
                    for method_frame, properties, body in channel.consume(
                            self.get_service_name(self.name), inactivity_timeout=self.timeout):
                        if method_frame is not None:
                            body = body.decode()
                            if body.startswith('ELEC'):
                                sender = body[4:]
                                channel.basic_publish(
                                    exchange='',
                                    routing_key=self.get_service_name(sender),
                                    body='OK')
                            else:
                                if body in workers:
                                    workers.remove(body)
                                if len(workers) == 0:
                                    channel.basic_ack(method_frame.delivery_tag)
                                    break
                            channel.basic_ack(method_frame.delivery_tag)
                        else:
                            break
                    for w in workers:
                        if w in self.workers:
                            self.workers.remove(w)
                            print(f'WORKER {w} DISCONNECTED')
                    for w in self.workers:
                        body = 'DEL' + SPLIT_TOKEN.join(workers)
                        channel.basic_publish(exchange='', routing_key=self.get_service_name(w), body=body)
                    time.sleep(self.timeout)
                else:
                    for method_frame, properties, body in channel.consume(
                            self.get_service_name(self.name), inactivity_timeout=7):
                        if method_frame is not None:
                            body = body.decode()
                            if body == 'PING':
                                channel.basic_publish(
                                    exchange='',
                                    routing_key=self.get_service_name(self.master_name),
                                    body=f'{self.name}'
                                )
                            elif body.startswith('DEL'):
                                workers = body[3:].split(SPLIT_TOKEN)
                                for w in workers:
                                    if w in self.workers:
                                        self.workers.remove(w)
                                        print(f'WORKER {w} DISCONNECTED')
                            elif body.startswith('COOR'):
                                self.master_name = body[4:]
                                print(f'>> {self.master_name} IS MASTER NOW')
                            channel.basic_ack(method_frame.delivery_tag)
                        elif method_frame is None or body.decode().startswith('ELEC'):
                            if method_frame is not None and body.decode().startswith('ELEC'):
                                sender = body[4:]
                                channel.basic_publish(
                                    exchange='',
                                    routing_key=self.get_service_name(sender),
                                    body='OK')
                                channel.basic_ack(method_frame.delivery_tag)
                            for w in self.workers:
                                if int(w) < int(self.name):
                                    channel.basic_publish(
                                        exchange='',
                                        routing_key=self.get_service_name(w),
                                        body=f'ELEC{self.name}'
                                    )
                            for method_frame_1, properties_1, body_1 in channel.consume(
                                    self.get_service_name(self.name), inactivity_timeout=6):
                                if method_frame_1 is not None:
                                    body_1 = body_1.decode()
                                    if body_1 == 'PING':
                                        channel.basic_publish(
                                            exchange='',
                                            routing_key=self.get_service_name(self.master_name),
                                            body=f'{self.name}'
                                        )
                                    elif body_1.startswith('DEL'):
                                        workers = body_1[3:].split(SPLIT_TOKEN)
                                        for w in workers:
                                            if w in self.workers:
                                                self.workers.remove(w)
                                                print(f'WORKER {w} DISCONNECTED')
                                    elif body_1.startswith("ELEC"):
                                        sender = body_1[4:]
                                        channel.basic_publish(
                                            exchange='',
                                            routing_key=self.get_service_name(sender),
                                            body='OK')
                                    elif body_1.startswith('COOR'):
                                        self.master_name = body_1[4:]
                                        print(f'>> {self.master_name} IS MASTER NOW')
                                    elif body_1.startswith('OK'):
                                        channel.basic_ack(method_frame_1.delivery_tag)
                                        break
                                    channel.basic_ack(method_frame_1.delivery_tag)
                                else:
                                    print(f'>> I AM MASTER {self.name}')
                                    self.master_name = self.name
                                    self.master = True
                                    for w in self.workers:
                                        channel.basic_publish(
                                            exchange='',
                                            routing_key=self.get_service_name(w),
                                            body=f'COOR{self.name}'
                                        )
                                    channel.basic_publish(
                                        exchange='',
                                        routing_key=self.name,
                                        body=f'AWAKE'
                                    )
                                    break
                        if self.master:
                            break
        finally:
            channel.queue_purge(queue=self.get_service_name(self.name))
            connection.close()

    def get_job_callback(self, channel, method, header, body):
        if self.service_thread is None:
            # self.service_thread = multiprocessing.Process(target=self.service)
            self.service_thread = threading.Thread(target=self.service)
            self.service_thread.start()
        body = body.decode()
        print(f'>> GET {body}')
        id, client_name, group_num, result_dir = tuple(body.split(SPLIT_TOKEN))
        if int(id) <= self.last_task:
            channel.basic_publish(exchange='', routing_key=frontend_queue_name,
                                  body=f'MAS{client_name}{SPLIT_TOKEN}{result_dir}')
            return
        shutil.rmtree(result_dir, ignore_errors=True)
        os.mkdir(result_dir)
        group_num = int(group_num)
        n, m = self.get_grid(group_num)
        x_step = (self.maxx - self.minx) / n
        y_step = (self.maxy - self.miny) / m
        self.wait_jobs = group_num
        while self.wait_jobs > 0:
            cur_worker = 0
            my_jobs = []
            self.wait_jobs = group_num
            for i in range(n):
                for j in range(m):
                    body = f'{i};{j}{SPLIT_TOKEN}'
                    body += f'{result_dir}{SPLIT_TOKEN}'
                    body += f'{self.minx + i * x_step};{self.miny + j * y_step}{SPLIT_TOKEN}'
                    body += f'{self.minx + (i + 1) * x_step};{self.miny + j * y_step}{SPLIT_TOKEN}'
                    body += f'{self.minx + (i + 1) * x_step};{self.miny + (j + 1) * y_step}{SPLIT_TOKEN}'
                    body += f'{self.minx + i * x_step};{self.miny + (j + 1) * y_step}{SPLIT_TOKEN}'
                    body += f'{self.minx + i * x_step};{self.miny + j * y_step}'
                    w = self.workers[cur_worker]
                    if w == self.name:
                        print(f'>> TAKE {body} JOB')
                        my_jobs.append(body)
                    else:
                        print(f'>> SEND {body} JOB TO {w}')
                        channel.basic_publish(exchange='', routing_key=w, body=body)
                    cur_worker = (cur_worker + 1) % len(self.workers)
            for job in my_jobs:
                self.process_job(job)
            self.wait_jobs -= len(my_jobs)
            print(f'>> WAIT JOBS {self.wait_jobs}')
            for method_frame, properties, body in channel.consume(self.name, inactivity_timeout=7):
                if method_frame is not None:
                    print(f'>> GET PIECE OF JOB')
                    self.wait_jobs -= 1
                    channel.basic_ack(method_frame.delivery_tag)
                    if self.wait_jobs == 0:
                        break
                else:
                    break

        channel.basic_publish(exchange='', routing_key=frontend_queue_name,
                          body=f'MAS{client_name}{SPLIT_TOKEN}{result_dir}')
        self.last_task = int(id)
        channel.stop_consuming()

    def worker_callback(self, channel, method, header, body):
        body = body.decode()
        print(f'>> GET {body}')
        if self.service_thread is None:
            # self.service_thread = multiprocessing.Process(target=self.service)
            self.service_thread = threading.Thread(target=self.service)
            self.service_thread.start()
        if body == 'AWAKE':
            channel.queue_declare(queue=frontend_queue_name)
            channel.queue_declare(queue=master_queue_name)
            channel.stop_consuming()
            return
        self.process_job(body)
        channel.basic_publish(exchange='', routing_key=self.master_name, body='WOR')

    def start(self):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost', heartbeat=0)
        )
        channel = connection.channel()

        channel.queue_declare(queue=self.name)
        channel.queue_purge(queue=self.name)

        if self.master:
            channel.queue_declare(queue=frontend_queue_name)
            channel.queue_purge(queue=frontend_queue_name)
            channel.queue_declare(queue=master_queue_name)
            channel.queue_purge(queue=master_queue_name)

        self.service_running = True

        print(f'>> WORKER {self.name} STARTED')
        print(f'>> {self.master_name} IS MASTER')

        try:
            while True:
                if self.master:
                    channel.basic_consume(
                        on_message_callback=self.get_job_callback,
                        queue=master_queue_name,
                        auto_ack=True
                    )
                    channel.start_consuming()
                else:
                    for method_frame, properties, body in channel.consume(self.name):
                        body = body.decode()
                        print(f'>> GET {body}')
                        if self.service_thread is None:
                            # self.service_thread = multiprocessing.Process(target=self.service, args=(self, ))
                            self.service_thread = threading.Thread(target=self.service)
                            self.service_thread.start()
                        if body == 'AWAKE':
                            print(f'>> {self.master} {self.master_name} {self.workers}')
                            channel.queue_declare(queue=frontend_queue_name)
                            channel.queue_declare(queue=master_queue_name)
                            break
                        self.process_job(body)
                        channel.basic_publish(exchange='', routing_key=self.master_name, body='WOR')
        finally:
            self.service_running = False
            self.service_thread.join()
            channel.queue_purge(queue=self.name)
            if self.master:
                channel.queue_purge(queue=master_queue_name)
            connection.close()
            print(f'WORKER {self.name} STOPPED')


if __name__ == '__main__':
    worker = Worker(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    worker.start()
