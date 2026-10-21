![alt text](https://github.com/raikustar/pytorch-tank/blob/main/prediction.png)
> {
    "label": "BMP3",
    "score": 84,
    "coordinates": [
        75,
        45,
        949,
        614
    ]
}

Trained YOLOv26 model on images of Russian tanks, BMP1/2 and BMP3. Will try to identify those 3 objects.

# **TBD**
In the process of getting updated.
### To do:
- improved model precision
- expand model dataset


## Usage
In order to setup the prototype-vison container the following are needed:  
Existing `RabbitMQ` container/localized service and some `backend framework` that has RabbitMQ configured.

`Backend framework` will send a *base64 binary string* as the converted image data to `RabbitMQ Message Broker` and that gets listened and consumed by the prototype-vision container, which will update and produce a response to the `Backend framework` that is constantly listening to the container as a consumer.

Flow of logic:  
`Backend [producer]` -> `RabbitMQ Message Broker` -> `prototype-vision[consumer]`  
`Backend [consumer]` <- `RabbitMQ Message Broker` <- `prototype-vision[producer]`

*.env.sample is in image/ - Contains values that are needed for the container configuration*  

### Container image  
Early stages: `docker pull ghcr.io/raikustar/prototype-vision:v1.02`

## Setup .env
This name should match what the rabbit-mq container service name is and not container_name. 
`RABBITMQ_HOST_NAME="message-broker"` 

The containers values as a consumer  
`RABBITMQ_CONSUMER_QUEUE="request.queue"`

The containers values as a producer  
`RABBITMQ_PRODUCER_QUEUE="response.queue"`  
`RABBITMQ_PRODUCER_EXCHANGE="response.exchange"`  
`RABBITMQ_PRODUCER_ROUTING="response.routing"`  

Dont use "guest", wont work in prod  
`RABBITMQ_USERNAME="custom-user"`   
`RABBITMQ_PASSWORD="custom-pw"`  

## In docker compose
>backend (with a rabbitmq producer and consumer configuration): 
   env_file:  
      - .env  
    environment:  
      - RABBITMQ_HOST_NAME=${RABBITMQ_HOST_NAME}  
      - RABBITMQ_DEFAULT_USER=${RABBITMQ_USER}  
      - RABBITMQ_DEFAULT_PASS=${RABBITMQ_PASSWORD}  
--- 
>message-broker:  
    container_name: rabbit-mq  
    image: rabbitmq:4-management  
    env_file:  
     - .env  
    environment:  
      - RABBITMQ_DEFAULT_USER=${RABBITMQ_USER}  
      - RABBITMQ_DEFAULT_PASS=${RABBITMQ_PASSWORD}  
    expose:  
      - "5672" #default rabbit-mq port  
--- 
>vision:  
    container_name: cv-python  
    image: ghcr.io/raikustar/prototype-vision:v1.02  
    env_file:  
      - .env  
    environment:  
      - RABBITMQ_HOST_NAME=${RABBITMQ_HOST_NAME}  
      - RABBITMQ_CONSUMER_QUEUE=${RABBITMQ_CONSUMER_QUEUE}  
      - RABBITMQ_PRODUCER_QUEUE=${RABBITMQ_PRODUCER_QUEUE}  
      - RABBITMQ_PRODUCER_EXCHANGE=${RABBITMQ_PRODUCER_EXCHANGE}  
      - RABBITMQ_PRODUCER_ROUTING=${RABBITMQ_PRODUCER_ROUTING}  
      - RABBITMQ_USER=${RABBITMQ_USER}  
      - RABBITMQ_PASSWORD=${RABBITMQ_PASSWORD}  

## Dataset
path: ../pytorch-tank/datasets  
train: data/train_data (90% of data)  
val: data/test_data (10% of data)  

**Labels:**
  >1: Tank  
  >2: BMP1/2  
  >3: BMP3  
  
