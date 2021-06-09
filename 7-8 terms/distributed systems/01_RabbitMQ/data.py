import pika
import xml.etree.ElementTree as et
from collections import defaultdict
import sys


def get_dict_from_OSM(file):
    d = defaultdict(set)
    tree = et.parse(file)
    for way in tree.findall('way'):
        highway = False
        name = ""
        for tag in way.findall('tag'):
            if tag.attrib['k'] == 'highway':
                highway = True
            if tag.attrib['k'] == 'name':
                name = tag.attrib['v'].lower()
            if highway and name != "":
                d[name[0]].add(name)

    return d


def callback(channel, method, header, body):
    print(f" [x] Requested streets starts with {body.upper()}")
    channel.basic_publish(exchange='', routing_key=sys.argv[1] + '_queue_d2m', body="|".join(streets_dict[body.decode()]))


streets_dict = get_dict_from_OSM(sys.argv[1] + ".osm")

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue=sys.argv[1] + '_queue_m2d')
channel.queue_declare(queue=sys.argv[1] + '_queue_d2m')

print(' [*] Waiting for messages. To exit press CTRL+C')

channel.basic_consume(on_message_callback=callback, queue=sys.argv[1] + '_queue_m2d', auto_ack=False)
try:
    channel.start_consuming()
except KeyboardInterrupt as e:
    channel.stop_consuming()
finally:
    channel.queue_delete(sys.argv[1] + '_queue_m2d')
    channel.queue_delete(sys.argv[1] + '_queue_d2m')
    connection.close()
