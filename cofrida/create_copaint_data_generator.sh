python3 create_copaint_data.py \
        --use_cache \
        --cofrida_dataset nateraw/parti-prompts \
        --cache_dir ../src/caches/sharpie_short_strokes \
        --materials_json ../materials_ink.json \
        --lr_multiplier 1.0  \
        --n_iters 700  \
        --max_strokes_added 400  \
        --min_strokes_added 10  \
        --ink \
        --render_height 168 \
        --output_parent_dir train_data_generator/ink \
        --generate_cofrida_training_data