"""
Main file.
"""

from os import getenv
from json import dumps
from binascii import Error as BinasciError
from dotenv import load_dotenv
from pika import (
    BlockingConnection,
    ConnectionParameters,
    PlainCredentials,
    BasicProperties,
)
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
    """
    Establises the RabbitMQ connection.
    """
    connection = BlockingConnection(
        ConnectionParameters(
            host=ML_HOST,
            virtual_host="/",
            credentials=PlainCredentials(ML_USER, ML_PASSWORD),
            connection_attempts=3,
            retry_delay=5,
        )
    )
    return connection.channel()


def processing_response(inner_channel, queue_id, status, result=None):
    """
    Helper function that is used to update the current state of the image inference result.

    `queue_id`: Current job ID

    `status`: Current state of the processing

    `result`: List of unique results from image inference
    """
    payload = {"id": queue_id, "status": status, "result": result}

    inner_channel.basic_publish(
        exchange=ML_RESPONSE_EXCHANGE,
        routing_key=ML_RESPONSE_ROUTING,
        properties=BasicProperties(correlation_id=queue_id),
        body=dumps(payload).encode(),
    )


def on_message(inner_channel, method, properties, body):
    """
    Message received from the queue.

    Image processing and detection inference with updating the message status.
    """
    try:
        # "processing image" -> stage 1 processing image
        processing_response(
            inner_channel, properties.correlation_id, "processing image"
        )
        ort_value, original_size = process_image(body.decode())

        # "detecting objects" -> stage 2 run detection
        processing_response(
            inner_channel, properties.correlation_id, "detecting objects"
        )
        response: list = detect_objects(
            MODEL_SESSION, ort_value, original_size, LABEL_NAMES, IMG_SIZE
        )

        # "completed" -> stage 3a return inference result
        processing_response(
            inner_channel, properties.correlation_id, "completed", response
        )
        inner_channel.basic_ack(delivery_tag=method.delivery_tag)
    except (Exception, TypeError, KeyError, ValueError, BinasciError) as exception:
        # "failed" -> stage 3b return failed with no result as data
        processing_response(inner_channel, properties.correlation_id, "failed")
        inner_channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


if __name__ == "__main__":
    channel = start_connection()

    channel.queue_declare(queue=ML_REQUEST_QUEUE, durable=True)
    channel.queue_declare(queue=ML_RESPONSE_QUEUE, durable=True)

    # acknowledgement(ack) is enabled by default. Messages are lost on crash.
    # If enabled: When a consumer receives the message, it's immediately marked as ack.
    # If disabled: 
    channel.basic_consume(
        queue=ML_REQUEST_QUEUE, on_message_callback=on_message, auto_ack=False
    )
    # Prefetch_count is an internal limit for RabbitMQ to stop forwarding messages to the consumer.
    # The consumer will always be limited to this number.
    channel.basic_qos(prefetch_count=3)

    print("Waiting for messages from producer")
    channel.start_consuming()
