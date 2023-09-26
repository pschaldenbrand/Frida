python3 create_copaint_data.py \
        --use_cache \
        --cache_dir ../src/caches/cache_6_6_cvpr \
        --lr_multiplier 0.7  \
        --n_iters 400  \
        --max_strokes_added 70  \
        --min_strokes_added 35  \
        --turn_takes 4  \
        --max_stroke_length 0.025 \
        --ink \
        --output_parent_dir train_data/ink