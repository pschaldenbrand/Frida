
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import argparse 
import os
import json

# Based off of Jun-Yan Zhu's Cyclegan implementation's options

class Options(object):
    def __init__(self):
        self.initialized = False
        self.opt = {}

        self.CANVAS_WIDTH_PIX  = None # set these after taking a picture
        self.CANVAS_HEIGHT_PIX = None

        self.HOVER_FACTOR = 0.1


    def initialize(self, parser):
        # Main parameters
        parser.add_argument("--robot", type=str, default='franka', help='Which robot to use "franka" "xarm" or "sawyer"')
        parser.add_argument("--xarm_ip", type=str, default='192.168.1.176', help='IP address of XArm.')

        parser.add_argument('--use_cache', action='store_true')
        parser.add_argument("--materials_json", type=str, 
            default='../materials.json', help='path to json file specifying material locations.')
        parser.add_argument("--cache_dir", type=str,
            default='./caches/cache_6_6_cvpr', help='Where to store cached files.')
        parser.add_argument('--simulate', action='store_true', help="Don't execute. Just plan without a robot. Requires already cached data.")
        parser.add_argument('--ink', action='store_true')
        parser.add_argument('--paint_from_image', action='store_true')

        # Color parameters
        parser.add_argument('--calib_colors', action='store_true', help='Use this to calibrate colors using MacBeth color checker')
        parser.add_argument('--n_colors', default=12, type=int, help='Number of colors of paint to use')
        parser.add_argument('--use_colors_from', type=str, default=None, help="Get the colors from this image. \
                None if you want the colors to come from the optimized painting.")
        parser.add_argument('--bin_size', type=int, default=3000)

        # Rendering Parameters
        parser.add_argument('--render_height', default=256, type=int, help='How much to downscale canvas for simulated environment')

        # Stroke Library Parameters
        parser.add_argument('--num_papers', default=4, type=int, help='How papers of strokes to paint for stroke modelling data.')
        parser.add_argument('--dont_retrain_stroke_model', action='store_true')
        parser.add_argument('--brush_length', type=float, default=None)

        # Planning Parameters
        parser.add_argument('--dont_plan', action='store_true', help='Use saved plan from last run')
        parser.add_argument('--num_strokes', type=int, default=400)
        parser.add_argument('--num_adaptations', type=int, default=1)
        parser.add_argument('--fill_weight', type=float, default=0.0, help="Encourage strokes to fill canvas.")

        # Optimization Parameters
        parser.add_argument('--objective', nargs='*', type=str, help='text|style|clip_conv_loss|l2|clip_fc_loss')
        parser.add_argument('--objective_data', nargs='*', type=str)
        parser.add_argument('--objective_weight', nargs='*', type=float, default=1.0)
        parser.add_argument('--init_optim_iter', type=int, default=400)
        parser.add_argument('--optim_iter', type=int, default=150)
        parser.add_argument('--lr_multiplier', type=float, default=0.2)

        parser.add_argument('--num_augs', type=int, default=30)

        # Painting Parameters
        parser.add_argument('--how_often_to_get_paint', type=int, default=4)

        # Logging Parameters
        parser.add_argument("--tensorboard_dir", type=str,
            default='./painting_log', help='Where to write tensorboard log to.')
        parser.add_argument('--plan_gif_dir', type=str, default='../outputs/')
        parser.add_argument('--log_frequency', type=int, default=5, help="Log to TB after this many optim. iters.")
        parser.add_argument("--output_dir", type=str, default="../outputs/", help='Where to write output to.')


        # CoFRIDA Run-Time Parameters
        parser.add_argument("--cofrida_model", type=str,
            default=None, help='path to pre-trained instruct-pix2pix CoFRIDA model')

        # CoFRIDA Training Parameters
        parser.add_argument("--cofrida_dataset", type=str,
            default="ChristophSchuhmann/MS_COCO_2017_URL_TEXT", help='A dataset for training CoFRIDA')
        parser.add_argument("--generate_cofrida_training_data", action='store_true',
            default=False, help='Generate training images for CoFRIDA. Use with dataset nateraw/parti-prompts')
        parser.add_argument("--cofrida_background_image", type=str,
            default='./blank_canvas.jpg', help='path to image to use as background for cofrida drawings/paintings')
        parser.add_argument("--output_parent_dir", type=str,
            help='Where to save the data. Can continue if partially complete.')
        # parser.add_argument("--removal_method", type=str,
        #     default='random',
        #     help='how to make partial sketchs. [random|salience]')
        parser.add_argument("--max_images", type=int,
            default=20000, help='A dataset for training controlnet')
        parser.add_argument("--num_images_to_consider_for_simplicity", type=int,
            default=3, help='Load this many images and take the one with fewest edges for simplicity.')
        parser.add_argument("--max_strokes_added", type=int,
            default=200, help='Maximum possible strokes in a cofrida training painting')
        parser.add_argument("--min_strokes_added", type=int,
            default=100, help='Minimum number of strokes in a cofrida training painting')
        parser.add_argument("--n_iters", type=int,
            default=300, help='Number of optimization iterations.')
        parser.add_argument("--colors", type=str,
            default=None, help='Specify a fixed palette of paint colors.')
        parser.add_argument("--codraw_metric_data_dir", type=str,
            default=None, help='Where to store evaluation data.')
        parser.add_argument("--codraw_eval_setting", type=str,
            default=None, help='[same_text_fill_in,same_text_add_detail_different_text,add_background,something_from_nothing]')
        parser.add_argument("--retrain_cofrida_image_generator", type=int,
            default=10, help='Retrain cofrida image generator every time this many images are generated.')
        
        
        ### Argument is not used, but is allowed for flask compatability ###
        parser.add_argument("--app", type=str, nargs='*',
            default="app run", help='Argument is not used, but is allowed for flask compatability')


        # Stroke Predictor Arguments
        parser.add_argument("--n_predicted_strokes", type=int,
            default=5, help='Number of strokes the stroke predictor should output')
        parser.add_argument("--n_predicted_strokes_low", type=int,
            default=None, help='Number of strokes the stroke predictor should output')
        parser.add_argument("--n_predicted_strokes_high", type=int,
            default=None, help='Number of strokes the stroke predictor should output')
        parser.add_argument("--sp_training_batch_size", type=int,
            default=24, help='Number of times to predict in a row')
        parser.add_argument("--n_gt_strokes", type=int,
            default=5, help='Number of strokes to add to target images')
        parser.add_argument("--continue_training", type=str,
            default=None, help='Path to a .pth model to continue training')
        parser.add_argument("--num_prediction_rounds", type=int,
            default=20, help='Number of times to predict in a row')
        parser.add_argument('--predict_many_then_optimize', action='store_true',
            help='If set, then predict --num_prediction_rounds of --n_predicted_strokes then optimize the final canvas.')
        parser.add_argument("--sp_lr", type=float,
            default=1e-4, help='initial learning rate for stroke predictor')
        parser.add_argument("--sp_no_pix_loss", action='store_true',
            help='')
        parser.add_argument("--sp_stroke_loss_weight", type=float,
            default=0.5, help='')
        parser.add_argument("--sp_cofrida_loss_weight", type=float,
            default=0.5, help='')
        parser.add_argument('--sp_predict_path', action='store_true',
            help='If set, then predict trajectory not the latent.')

        return parser 

    def gather_options(self):
        if not self.initialized:
            self.parser = argparse.ArgumentParser(description="FRIDA Robot Painter")
            self.parser = self.initialize(self.parser)

        self.opt = vars(self.parser.parse_args())

        if not os.path.exists(self.materials_json):
            raise Exception('Cannot find materials JSON file: ' + self.materials_json)
        with open(self.materials_json, 'r') as f:
            materials = json.load(f)
        
        self.opt = {**self.opt, **materials}

        thresh = 0.001 # 0.1cm tolerance on overshooting the canvas
        self.Y_CANVAS_MAX = self.CANVAS_POSITION[1] + self.CANVAS_HEIGHT_M - thresh
        self.Y_CANVAS_MIN = self.CANVAS_POSITION[1] + thresh
        self.X_CANVAS_MAX = self.CANVAS_POSITION[0] + self.CANVAS_WIDTH_M/2 - thresh
        self.X_CANVAS_MIN = self.CANVAS_POSITION[0] - self.CANVAS_WIDTH_M/2 + thresh

        if self.use_cache and os.path.exists(os.path.join(self.cache_dir, 'stroke_settings_during_library.json')):
            print("Retrieving settings from stroke library for consistency:")
            with open(os.path.join(self.cache_dir, 'stroke_settings_during_library.json'), 'r') as f:
                settings = json.load(f)
                print(settings)
                self.MAX_BEND = settings['MAX_BEND']
                self.MIN_STROKE_Z = settings['MIN_STROKE_Z']
                self.MIN_STROKE_LENGTH = settings['MIN_STROKE_LENGTH']
                self.MAX_STROKE_LENGTH = settings['MAX_STROKE_LENGTH']
                self.MAX_ALPHA = settings['MAX_ALPHA']
                self.STROKE_LIBRARY_CANVAS_WIDTH_M = settings['CANVAS_WIDTH_M']
                self.STROKE_LIBRARY_CANVAS_HEIGHT_M = settings['CANVAS_HEIGHT_M']

    def __getattr__(self, attr_name):
        return self.opt[attr_name]
