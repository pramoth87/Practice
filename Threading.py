# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 13:15:32 2021

@author: prchandr
"""

from tkinter import *
from threading import Thread

def scanning():
    while True:
        print ("hello")
        if stop == 1:   
            break   #Break while loop when stop = 1

def start_thread():
    # Assign global variable and initialize value
    global stop
    stop = 0

    # Create and launch a thread 
    t = Thread (target = scanning)
    t.start()

def stop_thread():
    # Assign global variable and set value to stop
    global stop
    stop = 1

def createUI():
    root = Tk()
    root.title("Title")
    root.geometry("500x500")
    
    app = Frame(root)
    app.grid()
    
    start = Button(app, text="Start Scan",command=start_thread)
    stop = Button(app, text="Stop",command=stop_thread)
    
    start.grid()
    stop.grid()


