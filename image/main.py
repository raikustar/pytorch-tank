from pika import (
    BlockingConnection,
    ConnectionParameters,
    PlainCredentials,
    BasicProperties,
)
from json import dumps
from os import getenv
from dotenv import load_dotenv
from vision import process_model, process_image, detect_objects

load_dotenv()
ML_HOST = getenv("RABBITMQ_HOST_NAME")

ML_REQUEST_QUEUE = getenv("RABBITMQ_CONSUMER_QUEUE")

ML_RESPONSE_QUEUE = getenv("RABBITMQ_PRODUCER_QUEUE")
ML_RESPONSE_EXCHANGE = getenv("RABBITMQ_PRODUCER_EXCHANGE")
ML_RESPONSE_ROUTING = getenv("RABBITMQ_PRODUCER_ROUTING")
ML_USER = getenv("RABBITMQ_USER")
ML_PASSWORD = getenv("RABBITMQ_PASSWORD")

MODEL_SESSION, IMG_SIZE, LABEL_NAMES = process_model()


def start_connection() -> BlockingConnection:
    return BlockingConnection(
        ConnectionParameters(
            host=ML_HOST,
            virtual_host="/",
            credentials=PlainCredentials(ML_USER, ML_PASSWORD),
            connection_attempts=3,
            retry_delay=5,
        )
    )


def processing_response(channel, queue_id, status, result=None):
    payload = {"id": queue_id, "status": status, "result": result}

    channel.basic_publish(
        exchange=ML_RESPONSE_EXCHANGE,
        routing_key=ML_RESPONSE_ROUTING,
        properties=BasicProperties(correlation_id=queue_id),
        body=dumps(payload).encode(),
    )


def on_message(channel, method, properties, body):

    # "processing image", -> stage 1 processing image
    processing_response(channel, properties.correlation_id, "processing image")
    ort_value, original_size = process_image(body.decode())

    # "detecting objects", -> stage 2 run detection
    processing_response(channel, properties.correlation_id, "detecting objects")
    response: list = detect_objects(
        MODEL_SESSION, ort_value, original_size, LABEL_NAMES, IMG_SIZE
    )

    # "completed"  or  "failed"
    processing_response(channel, properties.correlation_id, "completed", response)

    # TODO: There should be a failed response as well.

    channel.basic_ack(delivery_tag=method.delivery_tag)


if __name__ == "__main__":
    connection = start_connection()
    channel = connection.channel()

    channel.queue_declare(queue=ML_REQUEST_QUEUE, durable=True)
    channel.queue_declare(queue=ML_RESPONSE_QUEUE, durable=True)

    channel.basic_consume(queue=ML_REQUEST_QUEUE, on_message_callback=on_message)
    channel.basic_qos(prefetch_count=1)

    print("Waiting for messages from producer")
    channel.start_consuming()
