import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import pipeline
from transformers import AutoTokenizer, TFGPT2Tokenizer
from transformers import TFAutoModelForCausalLM, GenerationConfig, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import create_optimizer, AdamWeightDecay

import zipfile
import os
import math

checkpoint = "distilgpt2"
# checkpoint = "t5-small"
# checkpoint = "google/pegasus-xsum"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#tokenizer = TFGPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
#tf_tokenizer = TFGPT2Tokenizer.from_pretrained(tokenizer)
generator = pipeline(task="text-generation")
# task = "text2text-generation"

# IMPORTING DATASET

zip_ref = zipfile.ZipFile('alllines.txt.zip', 'r')
zip_ref.extractall('alllines')
zip_ref.close()
file_path = 'alllines/alllines.txt'
with open(file_path, 'r') as file:
    data = file.read()

lines = [line.rstrip() for line in open(file_path)]  # LIST OF ALL LINES
lines = [line for line in lines if len(line) > 0]

# PREPROCESSING OF DATA
import numpy as np

batch_size = 1000


def batch_iterator(dataset):
    for i in range(0, len(dataset), batch_size):
        return dataset[i: i + batch_size]


def tokenizer_function(inputs):
    return tokenizer(batch_iterator(inputs), padding=True, truncation=True)



tokenized_dataset = tokenizer_function(lines)
# print(tokenized_dataset)
# block_size = 128
block_size = 512


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result




# dataset = tokenized_dataset.map(
#     group_texts,
#     batched=True,
#     batch_size=1000,
#     num_proc=4,
# )
# dataset = group_texts(tokenized_dataset)


dataset = group_texts({key: tokenized_dataset[key] for i, key in enumerate(tokenized_dataset)})
# print(dataset['input_ids'])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="tf")  # dynamically pad the sentences to the longest length in a batch during collation
# small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))

## MODEL
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

model = TFAutoModelForCausalLM.from_pretrained("distilgpt2", checkpoint)
model.generation_config
values = list(dataset.values())

# Split the values into training and testing sets.
values_train, values_test = train_test_split(values, test_size=0.2, random_state=42)

# Create new dictionaries for the training and testing sets.
tf_train_set = {k: v for k, v in dataset.items() if v in values_train}
tf_test_set = {k: v for k, v in dataset.items() if v in values_test}

model.compile(optimizer=optimizer)
#for i,t in zip(range(len(tf_train_set['input_ids'])),range(len(tf_test_set['input_ids']))):
model.fit(tf.constant(tf_train_set['input_ids'], dtype=tf.float32),tf.constant(tf_test_set['input_ids'], dtype=tf.float32), epochs=3)
model.generation_config.max_length = 256
train_dataset = tf.constant(tf_train_set['input_ids'],dtype=tf.float32)
att_m = tf.constant(tf_train_set['attention_mask'],dtype=tf.int32)
attention_mask_dataset = tf.dtypes.cast(att_m, tf.float32)
output = model.generate(train_dataset,attention_mask = attention_mask_dataset, do_sample=True, num_beams=1, max_new_tokens=100)  # Multinomial sampling decoding
# output = model.generate(small_train_dataset, do_sample=True, num_beams=5, do_sample=True) # Beam-search multinomial sampling decoding
# output = model.generate(small_train_dataset, num_beams=5, num_beam_groups=5, max_new_tokens=30, diversity_penalty=1.0) # Diverse beam search decoding
tokenizer.batch_decode(output, skip_special_tokens=True)

eval_loss = model.evaluate(validation_set)

print(f"Perplexity: {math.exp(eval_loss):.2f}")