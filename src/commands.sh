python3 paint.py --use_cache --cache_dir caches/mars_brush --materials_json ../materials_mars_7x7.json  --objective clip_conv_loss --objective_data ~/Downloads/jj.jpg --objective_weight 1.0  --num_adaptations 1  --num_strokes 25 --init_optim_iter 400 --lr_multiplier 2 --ink --dont_retrain_stroke_model

python3 paint.py --use_cache --cache_dir caches/mars_brush_lift --materials_json ../materials_mars_7x7.json  --objective clip_conv_loss --objective_data ~/Downloads/uksang.jpg --objective_weight 1.0  --num_adaptations 1  --num_strokes 36 --init_optim_iter 800 --lr_multiplier 5  --dont_retrain_stroke_model --robot xarm --use_colors_from ~/Downloads/4grey.png --n_colors 4