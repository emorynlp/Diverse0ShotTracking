
import dextrous.launch.launch as launch


launch.LlamaLaunch(
    approach='LlamaTracker',
    train_data='data/dsg5k_filtered/train',
    eval_data='data/mwoz2.4/valid',
    train_downsample=None,
    eval_dialogue_downsample=10,
    eval_exclude_speakers='bot',
    test_domains=None,
    eval_all_slots_per_domain=True,
    prediction_lowercase=True,
    prediction_fuzzy_match_candidates=True,
    epochs=1,
    max_sequence_length=1024,
    param_magnitude='13b',
    train_batch_size=256,
    gradient_accumulation_steps=128,
    warmup_steps=100,
    learning_rate=5e-5,
    weight_decay=0.0,
    quantize='nf4',
    lora=32,
    lora_alpha=64,
    lora_dropout=0.0,
    neg_examples_ratio=0.5,
    train_prop_add_continuation=None,
    train_prop_add_continuation_pcent_existing=None,
    train_continuation_exs_replace_original=True,
    train_percent_with_description=0.9,
    train_percent_description_only=0.1,
    train_percent_with_categories=0.25,
    train_percent_with_value_exs=0.75,
    train_percent_value_ex_includes_actual_value=0.1,
    train_remove_request_token_percent=1.0,
    train_filters_out_descriptions_with_actual_value=None,
    eval_with_categories=False,
    eval_with_value_exs=False,
    uncased=False,
    num_beams=3,
    repetition_penalty=1.0,
    gen_batch_size=6,
    max_output_length=16,
    rng_seed=21,
    do_eval_after_all_training=False,
    calculate_eval_perplexity=False,
    yield_every_x_epochs=0.05,
    dynamic_tokenization=True,
    notifications=True,
)