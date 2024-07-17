
##########################################################
#################### Copyright 2023 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import datetime
import os
import sys
import time
import easygui
import torch
from torchvision.transforms import Resize
from tqdm import tqdm
from pynput import keyboard
from IPython import embed

from paint_utils3 import canvas_to_global_coordinates, format_img
from create_plan import define_prompts_dictionary

from painter import Painter
from options import Options
from my_tensorboard import TensorBoard

# Adding ROS Imports
import roslib, sys, rospy, os
from std_msgs.msg import String, Int32

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

def define_performance_mapping():
    
    global performance_dictionary    
    performance_dictionary = {}
    performance_dictionary['1'] = 'Good'
    performance_dictionary['0'] = 'Medium'

    return performance_dictionary

def retrieve_queue_performance():
    # Retrieve the perofrmance value.
    return performance_dictionary[str(performance_queue.pop(0))]

def flip_img(img):
    # Utility
    return torch.flip(img, dims=(2,3))

def execute_painting(painting):
    # Execute plan
    n_strokes = 50
    
    # for stroke_ind in tqdm(range(len(painting)), desc="Executing plan"):
    for stroke_ind in tqdm(range(n_strokes), desc="Executing plan"):
        while is_paused:
            time.sleep(0.1)

        stroke = painting.pop()

        # Convert the canvas proportion coordinates to meters from robot
        x, y = stroke.xt.item(), stroke.yt.item()
        y = 1-y
        x, y = min(max(x,0.),1.), min(max(y,0.),1.) #safety
        x_glob, y_glob,_ = canvas_to_global_coordinates(x,y,None,painter.opt)

        # Runnit
        stroke.execute(painter, x_glob, y_glob, stroke.a.item(), fast=True)

        if opt.simulate:
            time.sleep(1)

    painter.to_neutral()

is_paused = False

if __name__ == '__main__':

    if True:
        ############################
        # Setting parameters. 
        ############################

        opt = Options()
        opt.gather_options()

        date_and_time = datetime.datetime.now()
        run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
        opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
        opt.writer.add_text('args', str(sys.argv), 0)

        painter = Painter(opt)
        opt = painter.opt 

        painter.to_neutral()

        w_render = int(opt.render_height * (opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))
        h_render = int(opt.render_height)
        opt.w_render, opt.h_render = w_render, h_render

        ############################
        # Defining logic for pausing drawing during exercise. 
        ############################

        print('Press any key to pause/continue')

    def on_press(key):
        try:
            global is_paused

            if key.char == keyboard.Key.f1:
                is_paused = True
            if key.char == keyboard.Key.f1:
                is_paused = False

            # is_paused = not is_paused
            # print('alphanumeric key {0} pressed'.format(
            #     key.char))

            if is_paused:
                print("Paused")
            else:
                print('Resuming')
        except AttributeError:
            # print('special key {0} pressed'.format(
            #     key))
            print('some error')    

    ############################
    # Initialize ROS NOde
    ############################

    rospy.init_node('exercise_performance_subscriber', anonymous=True)

    ############################
    # Defining Prompts
    ############################

    # Generate the prompts dictionary. 
    prompt_dict = define_prompts_dictionary()
    define_performance_mapping()
    
    # Defining global performance variable. 
    global performance_queue    
    performance_queue = []
    
    ############################
    # Get input for which of the 10 prompts we want to start drawing with. 
    ############################

    # plan_dir_index = int(input('Which prompt number should I draw? Please enter a number from 0 to 9.'))
    plan_dir_index = 1
    # ...or, in a non-blocking fashion:
    listener = keyboard.Listener(
        on_press=on_press)
    # listener.start()
    
    save_dir = opt.saved_plan
    plan_dir = os.path.join(save_dir, 'Painting{}'.format(plan_dir_index))

    ############################
    # Visualize the planned painting.
    ############################

    # TODO

    ############################
    # Run Planning for Initial Prompt. 
    ############################

    # Load plan from saved directory. 
    init_painting_plan = torch.load(os.path.join(plan_dir, 'InitialPrompt', 'plan.pt'))   

    # Execute this plan. 
    execute_painting(init_painting_plan)

    ############################
    # Get input for whether to run the Medium branch or the Good Branch of this tree of prompts. 
    ############################

    # subsequent_plan_branch = int(input('How well did the user perform their exercise? Please enter either "Good" or "Medium". This will determine which branch of the tree I will draw.'))        
    performance_measure = rospy.wait_for_message("/set_performance", Int32, timeout=None)
    # print("received message")

    print("Received Message ", performance_measure, performance_measure.data)
    
    # embed()
    subsequent_plan_branch = performance_dictionary[str(performance_measure.data)]
    print("The plan branch we are going to execute is: ", subsequent_plan_branch)
    # # subsequent_plan_branch = retrieve_queue_performance()
    # subsequent_plan_dir = os.path.join(save_dir, 'Painting{}'.format(subsequent_plan_branch))
    
    # subsequent_plan_dir = os.path.join(plan_dir, '{}'.format(subsequent_plan_branch))

    ############################
    # Visualize the planned painting.
    ############################

    # TODO

    ############################
    # Run Planning for Subsequent Prompt.
    ############################

    # Load plan from saved directory. 
    prompt_key = subsequent_plan_branch+"SubsequentPrompt"
    subsequent_painting_plan = torch.load(os.path.join(plan_dir, prompt_key, 'plan.pt'))

    # Execute this plan. 
    execute_painting(subsequent_painting_plan)

    ############################
    # Logging if we want.
    ############################

    current_canvas = painter.camera.get_canvas_tensor() / 255.
    current_canvas = flip_img(current_canvas)
    opt.writer.add_image('images/{}_4_canvas_after_drawing'.format(0), format_img(current_canvas), 0)
    current_canvas = Resize((h_render, w_render), antialias=True)(current_canvas)
    
    ############################
    # Shutdown. 
    ############################

    painter.to_neutral()
    painter.robot.good_night_robot()