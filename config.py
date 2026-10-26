model_checkpoint = "lycanthropeus/mt5-small-finetuned-en-to-ar"
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "ar"
prefix = ""



# Seq2SeqTrainingArguments
eval_strategy = "epoch"
learning_rate = 1e-3
optim = "adafactor"
warmup_steps = 500
weight_decay = 0.0
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
gradient_accumulation_steps = 2
num_train_epochs = 3
fp16 = False
predict_with_generate = True
generation_max_length = 128
save_total_limit = 3
push_to_hub = True


#eval.py
trained_model_checkpoint = "lycanthropeus/mt5-small-finetuned-en-to-ar-200k"
N_EVAL = 100
MAX_WORKERS = 8