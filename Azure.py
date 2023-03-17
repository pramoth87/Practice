#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 14:07:00 2021

@author: pchandrasekar
"""
import json
import asyncio
from azure.eventhub.aio import EventHubProducerClient
from azure.eventhub import EventData

async def send_Events_To_Azure(message):
    # Create a producer client to send messages to the event hub.
    # Specify a connection string to your event hubs namespace and
    # the event hub name.
    producer = EventHubProducerClient.from_connection_string(
        conn_str="Endpoint=sb://azm-logs-poc-eh.servicebus.windows.net/;SharedAccessKeyName=irwin-send;SharedAccessKey=TYhzAUoinDjm+9ZGKcqforFc6zzI6md9lyA54Umtj1c=;EntityPath=aad-logs",
        eventhub_name="aad-logs",
        consumer_group="python")
    async with producer:
        # Create a batch.
        event_data_batch = await producer.create_batch()
        # Add events to the batch.
        event_data_batch.add(EventData(message))
        # Send the batch of events to the event hub.
        await producer.send_batch(event_data_batch)

loop = asyncio.get_event_loop()
with open('data/aad.json') as json_file:
    aadLog = json.load(json_file)
loop.run_until_complete(send_Events_To_Azure(aadLog))



"""
Formula one race you have n laps to finish. Every lap takes t secs after each lap the the tire gets worn out so the lap will 
take t * f (Degradation). You can choose to take the pit stop any time with p secs will be added to total time

1 Lap = t
2 Lap = t * f
3 Lap = t * f^2
4 Lap = t * f^3
if taken a pit stop it will lap starts again = t

n = 2
f = 2
p = 10 secs
t = 300 secs

300 + 300+ 10
300 + 300 * 2 + 300 * 4 = 300(1+x+x^2)

With Pit Stop = 610
without pit Stop = 900
"""

def totalTime(n,f,p,t):
    