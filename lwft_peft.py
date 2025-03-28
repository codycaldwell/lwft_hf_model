from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import AutoPeftModelForSequenceClassification, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import numpy as np
 
my_model_name = "distilbert-base-uncased"
 
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}
 
# Criteria #1. Load a pretrained HF model
model = AutoModelForSequenceClassification.from_pretrained(
    my_model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

 
# Create a tokenizer
tokenizer = AutoTokenizer.from_pretrained(my_model_name)
 
 
# Dataset to be used: Amazon reviews of mobile electronics
my_dataset_name = "rkf2778/amazon_reviews_mobile_electronics"
splits = ["test", "train"]
 
# Criteria #2a Load dataset
datasets = {
    split: dataset
    for split, dataset in zip(splits, load_dataset(my_dataset_name, split=splits))
}
 
# Reduce the number of data for efficiency
for split in splits:
    datasets[split] = datasets[split].shuffle(seed=50).select(range(500))
 
 
train_dataset = datasets["train"]
test_dataset = datasets["test"]
 
# Criteria 2b: Preprocess the dataset inlcuding the star_rating_label representing the sentiment
def preprocess_review(batch):
    tokenized = tokenizer(batch["review_body"], padding=True, truncation=True, max_length=512)
    tokenized["labels"] = [label2id[label] for label in batch["star_rating_label"]]
    return tokenized
 
# Tokenize the datasets
tokenized_train = train_dataset.map(preprocess_review, batched=True)
tokenized_test = test_dataset.map(preprocess_review, batched=True)
 
 
# Accuracy calculation function and Trainer setup is from Udacity lesson 4.14
def calculate_accuracy(predictions):
    logits, labels = predictions
    predicted_class = np.argmax(logits, axis=-1)
    return {"accuracy": (predicted_class == labels).mean()}
 
 
training_args = TrainingArguments(
    output_dir="./data/original_model_sentiments_output",
    per_device_eval_batch_size=16
)
 
my_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
 
# Create a trainer (referenced from Udacity lession 4.14)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=my_data_collator,
    compute_metrics=calculate_accuracy
)
 
 
# Criteria #3: Evaluate the untuned model
trainer.evaluate()
 
 
# Reference: https://huggingface.co/docs/peft/package_reference/lora
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"], # Layers specific to distilbert
    lora_dropout=0.1,
    task_type=TaskType.SEQ_CLS
)
 
# Criteria #4: Create a PEFT model
peft_model = get_peft_model(model, peft_config)
 
for param in peft_model.parameters():
    param.requires_grad = True
 
peft_model.print_trainable_parameters()
 
 
# Criteria #5: Train the PEFT model (source: Udacity lesson 4.14)
peft_training_args = TrainingArguments(
    output_dir="./data/peft_model_sentiments_output_2",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
)
 
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=my_data_collator,
    compute_metrics=calculate_accuracy
)
 
peft_trainer.train()
 
# Criteria #6: Save the PEFT model 
peft_model.save_pretrained("my_peft_lora_distilbert2")
 
 
# Criteria #7: Load the saved PEFT model (https://huggingface.co/docs/peft/package_reference/auto_class)
saved_peft_model = AutoPeftModelForSequenceClassification.from_pretrained("my_peft_lora_distilbert2")
 
# Recreate the PEFT trainer from the saved model
tuned_trainer = Trainer(
    model=saved_peft_model,
    args=peft_training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=my_data_collator,
    compute_metrics=calculate_accuracy
)
 
 
# Criteria #8: Compare and evaluate the fine tuned model
print(f"Original Model accuracy: {trainer.evaluate()}")
print(f"Fine-tuned Model accuracy: {tuned_trainer.evaluate()}")