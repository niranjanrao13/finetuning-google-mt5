from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from config import (
    model_checkpoint, 
    max_input_length,
    max_target_length, 
    source_lang, 
    target_lang, 
    prefix,
    per_device_eval_batch_size,
    per_device_train_batch_size,
    predict_with_generate,
    eval_strategy,
    learning_rate,
    optim,
    warmup_steps,
    weight_decay,
    generation_max_length,
    save_total_limit,
    push_to_hub,
    num_train_epochs,
    fp16,
    gradient_accumulation_steps,
    trained_model_checkpoint
)


def load_data():
    ds = load_dataset('Helsinki-NLP/opus-100', 'ar-en')
    train = ds["train"].select(range(min(200000, len(ds["train"]))))
    return DatasetDict({
        'train': train,
        'val':   ds['validation'],
        'test':  ds['test']
    })


def build_tokenizer():
    return AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess(dataset, tokenizer):
    def preprocess_function(examples):
        inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]

        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_input_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1000,
        remove_columns=['translation'],
        writer_batch_size=1000,
        desc="Tokenizing"
    )


def build_model():
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


def build_training_args():
    model_name = model_checkpoint.split("/")[-1]
    return Seq2SeqTrainingArguments(
        trained_model_checkpoint,
        eval_strategy=eval_strategy,
        learning_rate=learning_rate,
        optim=optim,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        fp16=fp16,
        predict_with_generate=predict_with_generate,
        generation_max_length=generation_max_length,
        save_total_limit=save_total_limit,
        push_to_hub=push_to_hub,
    )


def train():
    dataset   = load_data()
    tokenizer = build_tokenizer()
    tokenized = preprocess(dataset, tokenizer)
    model     = build_model()
    args      = build_training_args()

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    train()