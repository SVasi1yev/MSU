import pika


def portland_callback(channel, method, header, body):
    global portland_streets
    print(f" [x] Received Portland streets")
    portland_streets = body.decode().split("|")
    if portland_streets[0] == '':
        portland_streets = []
    channel.stop_consuming()


def jackson_lake_callback(channel, method, header, body):
    global jackson_lake_streets
    print(f" [x] Received Jackson Lake streets")
    jackson_lake_streets = body.decode().split("|")
    if jackson_lake_streets[0] == '':
        jackson_lake_streets = []
    channel.stop_consuming()


connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='portland_queue_m2d')
channel.queue_declare(queue='jackson_lake_queue_m2d')
channel.queue_declare(queue='portland_queue_d2m')
channel.queue_declare(queue='jackson_lake_queue_d2m')

portland_streets = None
jackson_lake_streets = None
try:
    # method, properties, body = channel.basic_get("portland_queue_d2m", portland_callback)
    # method, properties, body = channel.basic_get("jackson_lake_queue_d2m", jackson_lake_callback)
    print('start')
    while True:
        command = input()
        if command == "finish":
            break
        elif len(command) == 1:
            channel.basic_publish(exchange='', routing_key='portland_queue_m2d', body=command)
            channel.basic_publish(exchange='', routing_key='jackson_lake_queue_m2d', body=command)

            # method, properties, body = channel.basic_get("portland_queue_d2m", portland_callback)
            # method, properties, body = channel.consume(queue="portland_queue_d2m")

            channel.basic_consume(on_message_callback=portland_callback, queue='portland_queue_d2m', auto_ack=False)
            channel.start_consuming()
            print(f"Portland has {len(portland_streets)} streets starts with {command.upper()}")
            print(portland_streets)

            # method, properties, body = channel.basic_get("jackson_lake_queue_d2m", jackson_lake_callback)
            # method, properties, body = channel.consume(queue="jackson_lake_queue_d2m")

            channel.basic_consume(on_message_callback=jackson_lake_callback, queue='jackson_lake_queue_d2m', auto_ack=False)
            channel.start_consuming()
            print(f"Jackson Lake has {len(jackson_lake_streets)} starts with {command.upper()}")
            print(jackson_lake_streets)
            if len(portland_streets) > len(jackson_lake_streets):
                print('--- more streets in PORTLAND')
            elif len(jackson_lake_streets) > len(portland_streets):
                print('--- more streets in JACKSON LAKE')
            else:
                print('--- EQUAL NUMBER OF STREETS')
        else:
            print("wrong command!")
finally:
    connection.close()
