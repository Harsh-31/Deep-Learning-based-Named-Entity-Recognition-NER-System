# -*- coding: utf-8 -*-
"""
Deep Learning-based Named Entity Recognition (NER) System
"""

import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # for more reproducible CUDA behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

"""### **Prepare Dataset**"""

def read_file(file_path):
  sentences = []
  words = []
  tags = []

  with open(file_path, 'r') as file:
    for line in file:
      line = line.strip()
      if line == "":
        if words:
          sentences.append((words, tags))
          words = []
          tags = []
      else:
        divide_line_in_parts = line.split()
        word = divide_line_in_parts[1]
        tag = divide_line_in_parts[2]
        words.append(word)
        tags.append(tag)

  if words:
    sentences.append((words, tags))

  return sentences

file_path = "/content/train"
train_data = read_file(file_path)
print(train_data[0])

"""# **Build Vocabulary**"""

def build_vocabulary(sentences):
  word_to_index = {"<PAD>": 0, "<UNK>": 1}
  tag_to_index = {"<PAD>": 0}
  next_id_words = 2
  next_id_tags = 1

  for words, tags in sentences:
    for word in words:
      if word not in word_to_index:
        word_to_index[word] = next_id_words
        next_id_words = next_id_words + 1

  for words, tags in sentences:
    for tag in tags:
      if tag not in tag_to_index:
        tag_to_index[tag] = next_id_tags
        next_id_tags = next_id_tags + 1

  return word_to_index, tag_to_index

word_to_index, tag_to_index = build_vocabulary(train_data)
print(word_to_index)
print(tag_to_index)
print(list(word_to_index.items())[:10])
print("EU" in word_to_index)
print("rejects" in word_to_index)

"""# **Convert Train Data to Numbers**"""

def data_to_number(sentences, word_to_index, tag_to_index):
  encoded_sentences = []
  for words,tags in sentences:
    word_ids = []
    tag_ids = []

    for i in range(len(words)):
      word = words[i]
      tag = tags[i]
      if word in word_to_index:
        word_ids.append(word_to_index[word])
      else:
        word_ids.append(word_to_index["<UNK>"])

      tag_ids.append(tag_to_index[tag])

    encoded_sentences.append((word_ids, tag_ids))

  return encoded_sentences

train_encoded_data = data_to_number(train_data, word_to_index, tag_to_index)
print(train_encoded_data[0])

"""# **Dataset + Padding + Batching**"""

class NamedEntityRecognitionDataset(Dataset):
    def __init__(self, encoded_sentences):
        self.encoded_sentences = encoded_sentences

    def __len__(self):
        return len(self.encoded_sentences)

    def __getitem__(self, index):
        return self.encoded_sentences[index]


def pad_batch(batch, pad_word_id=0, pad_tag_id=0):
    word_sequences = []
    tag_sequences = []
    lengths = []

    for words, tags in batch:
        word_sequences.append(words)
        tag_sequences.append(tags)
        lengths.append(len(words))

    max_len = max(lengths)

    padded_words = []
    padded_tags = []

    for i in range(len(word_sequences)):
        words = word_sequences[i]
        tags = tag_sequences[i]

        word_padding = [pad_word_id] * (max_len - len(words))
        tag_padding = [pad_tag_id] * (max_len - len(tags))

        padded_words.append(words + word_padding)
        padded_tags.append(tags + tag_padding)

    return (
        torch.tensor(padded_words, dtype=torch.long),
        torch.tensor(padded_tags, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long)
    )


train_dataset = NamedEntityRecognitionDataset(train_encoded_data)
set_seed(42)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=pad_batch
)

for batch_words, batch_tags, batch_lengths in train_loader:
    print("Words shape:", batch_words.shape)
    print("Tags shape:", batch_tags.shape)
    print("Lengths:", batch_lengths)
    print("First sentence words:", batch_words[0])
    print("First sentence tags:", batch_tags[0])
    break

"""# **Build BiLSTM Model**"""

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=256, pad_index=0):
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.33
        )

        self.linear = nn.Linear(hidden_dim * 2, 128)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(128, tagset_size)

    def forward(self, words, lengths):
        embedded = self.embedding(words)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed)

        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )

        x = self.linear(output)
        x = self.elu(x)
        logits = self.classifier(x)

        return logits

vocab_size = len(word_to_index)
tagset_size = len(tag_to_index)
set_seed(42)
model = BiLSTM(vocab_size, tagset_size)

for batch_words, batch_tags, batch_lengths in train_loader:
    logits = model(batch_words, batch_lengths)
    print(logits.shape)
    break

"""# **Training**"""

def train_model(model, train_loader, dev_loader, tag_to_index, epochs = 15, learning_rate = 0.05, device = "cpu"):
  # Step-1: Set model to correct device
  model = model.to(device)

  # Step-2: Choose a loss function
  loss_function = nn.CrossEntropyLoss(ignore_index = tag_to_index["<PAD>"])

  # Step-3: Choose an Optimizer
  optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)

  # Step-4: Repeat training for epochs
  for epoch in range(epochs):

    model.train()
    total_loss = 0.0

    # Step-5: Loop over one batch at a time
    for batch_words, batch_tags, batch_lengths in train_loader:

      # Step-6: Move batch data to cpu/gpu
      batch_words = batch_words.to(device)
      batch_tags = batch_tags.to(device)
      batch_lengths = batch_lengths.to(device)

      # Step-7: Clear all gradients from previous batch
      optimizer.zero_grad()

      # Step-8: Forward Pass
      logits = model(batch_words, batch_lengths)

      # Step-9: Flatten shapes so CrossEntropyLoss can read them
      logits = logits.reshape(-1, logits.shape[-1])
      true_data = batch_tags.reshape(-1)

      # Step-10: Compute how wrong the modle is
      loss = loss_function(logits, true_data)

      # Step-11: Backward Pass
      loss.backward()

      # Step-12: Update the model parameters
      optimizer.step()

      # Step-13: Total Loss
      total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

"""# **Create Dev Loader**"""

# read dev data
dev_data = read_file("/content/dev")

# encode dev data using the same vocab as training
dev_encoded = data_to_number(dev_data, word_to_index, tag_to_index)

# wrap dev data in Dataset
dev_dataset = NamedEntityRecognitionDataset(dev_encoded)

# create DataLoader for dev
dev_loader = DataLoader(
    dev_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=pad_batch
)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)
train_model(
    model,
    train_loader,
    dev_loader,
    tag_to_index,
    epochs=15,
    learning_rate=0.2,
    device=device
)

"""# **Evaluation**"""

def evaluate_model(model, data_loader, device="cpu"):
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch_words, batch_tags, batch_lengths in data_loader:
            batch_words = batch_words.to(device)
            batch_tags = batch_tags.to(device)

            logits = model(batch_words, batch_lengths)

            # take the highest-scoring tag for each token
            preds = torch.argmax(logits, dim=-1)

            # go sentence by sentence
            for i in range(len(batch_lengths)):
                length = batch_lengths[i].item()

                # keep only the real tokens, not padding
                pred_seq = preds[i][:length].tolist()
                true_seq = batch_tags[i][:length].tolist()

                all_preds.append(pred_seq)
                all_true.append(true_seq)

    return all_preds, all_true

device = "cuda" if torch.cuda.is_available() else "cpu"

dev_preds, dev_gold = evaluate_model(model, dev_loader, device=device)

# precision, recall, f1 = compute_metrics(
#     dev_preds,
#     dev_gold,
#     pad_tag_id=tag_to_index["<PAD>"]
# )

# print("Dev Precision:", precision)
# print("Dev Recall:", recall)
# print("Dev F1:", f1)

"""# **Save Dev predictions to dev1.out**"""

def write_predictions_to_file(output_file, raw_sentences, predicted_seqs, tag_to_index):
    id_to_tag = {}
    for tag in tag_to_index:
        tag_id = tag_to_index[tag]
        id_to_tag[tag_id] = tag

    with open(output_file, "w") as f:
        for sent_idx in range(len(raw_sentences)):
            words, _ = raw_sentences[sent_idx]
            pred_seq = predicted_seqs[sent_idx]

            for i in range(len(words)):
                word = words[i]
                tag_id = pred_seq[i]
                tag = id_to_tag[tag_id]

                # same format as training data: index word tag
                f.write(f"{i+1} {word} {tag}\n")

            # blank line after each sentence
            f.write("\n")

write_predictions_to_file("dev1.out", dev_data, dev_preds, tag_to_index)

# python eval.py -g dev -p dev1.out

"""# **Prepare Test Data**"""

def read_file(file_path, has_tags=True):
    sentences = []
    words = []
    tags = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if line == "":
                if words:
                    if has_tags:
                        sentences.append((words, tags))
                    else:
                        sentences.append((words, None))
                    words = []
                    tags = []
            else:
                parts = line.split()
                word = parts[1]
                words.append(word)

                if has_tags:
                    tag = parts[2]
                    tags.append(tag)

    if words:
        if has_tags:
            sentences.append((words, tags))
        else:
            sentences.append((words, None))

    return sentences

"""# **Test Data and Test Loader**"""

test_data = read_file("/content/test", has_tags=False)

test_sentences = []
for words, _ in test_data:
    dummy_tags = ["O"] * len(words)
    test_sentences.append((words, dummy_tags))

test_encoded = data_to_number(test_sentences, word_to_index, tag_to_index)

test_dataset = NamedEntityRecognitionDataset(test_encoded)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=pad_batch
)

"""# **Predict Test Tags and save it in test1.out**"""

test_preds, _ = evaluate_model(model, test_loader, device=device)
write_predictions_to_file("test1.out", test_data, test_preds, tag_to_index)

"""# **TASK - 2: Using GloVe Word Embeddings**"""

def get_case_id(word):
    if word.islower():
        return 0

    elif word.isupper():
        return 1

    elif word[0].isupper() and word[1:].islower():
        return 2

    elif any(char.isdigit() for char in word):
        return 4

    else:
        return 3

"""# **Convert Data to Numbers Including Case Id**"""

def data_to_number_task2(sentences, word_to_index, tag_to_index):
  encoded_sentences = []
  for words,tags in sentences:
    word_ids = []
    tag_ids = []
    case_ids = []

    for i in range(len(words)):
      word = words[i]
      tag = tags[i]
      if word in word_to_index:
        word_ids.append(word_to_index[word])
      else:
        word_ids.append(word_to_index["<UNK>"])

      tag_ids.append(tag_to_index[tag]) # tag ids
      case_ids.append(get_case_id(word)) # case ids

    encoded_sentences.append((word_ids, case_ids, tag_ids))

  return encoded_sentences

"""# **Encoded Train/Dev Data**"""

train_encoded_task2 = data_to_number_task2(train_data, word_to_index, tag_to_index)
dev_encoded_task2 = data_to_number_task2(dev_data, word_to_index, tag_to_index)

"""# **Dataset + Padding + Batching for Task-2**"""

class NamedEntityRecognitionDataset_Task2(Dataset):
    def __init__(self, encoded_sentences):
        self.encoded_sentences = encoded_sentences

    def __len__(self):
        return len(self.encoded_sentences)

    def __getitem__(self, index):
        return self.encoded_sentences[index]


def pad_batch_task2(batch, pad_word_id=0, pad_case_id=0, pad_tag_id=0):
    word_sequences = []
    case_sequences = []
    tag_sequences = []
    lengths = []

    for words, cases, tags in batch:
        word_sequences.append(words)
        case_sequences.append(cases)
        tag_sequences.append(tags)
        lengths.append(len(words))

    max_len = max(lengths)

    padded_words = []
    padded_tags = []
    padded_cases = []

    for i in range(len(word_sequences)):
        words = word_sequences[i]
        tags = tag_sequences[i]
        cases = case_sequences[i]

        word_padding = [pad_word_id] * (max_len - len(words))
        case_padding = [pad_case_id] * (max_len - len(cases))
        tag_padding = [pad_tag_id] * (max_len - len(tags))

        padded_words.append(words + word_padding)
        padded_cases.append(cases + case_padding)
        padded_tags.append(tags + tag_padding)

    return (
        torch.tensor(padded_words, dtype=torch.long),
        torch.tensor(padded_cases, dtype=torch.long),
        torch.tensor(padded_tags, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long)
    )


train_dataset = NamedEntityRecognitionDataset_Task2(train_encoded_task2)
train_loader_task2 = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=pad_batch_task2
)

for batch_words, batch_cases, batch_tags, batch_lengths in train_loader_task2:
    print("Words shape:", batch_words.shape)
    print("Cases shape:", batch_cases.shape)
    print("Tags shape:", batch_tags.shape)
    print("Lengths:", batch_lengths)
    print("First sentence words:", batch_words[0])
    print("First sentence cases:", batch_cases[0])
    print("First sentence tags:", batch_tags[0])
    break

"""# **Build BiLSTM Model for Task-2**"""

class BiLSTM_Task2(nn.Module):
    def __init__(self, vocab_size, tagset_size, word_embedding_dim=100, case_embedding_dim=10, hidden_dim=256, pad_index=0, num_case_types=5):
        super(BiLSTM_Task2, self).__init__()

        # word embedding
        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=word_embedding_dim,
            padding_idx=pad_index
        )

        # case embedding layer
        self.case_embedding = nn.Embedding(
            num_embeddings=num_case_types,
            embedding_dim=case_embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=word_embedding_dim + case_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.33
        )

        self.linear = nn.Linear(hidden_dim * 2, 128)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(128, tagset_size)

    def forward(self, words, cases, lengths):
        word_embeds = self.word_embedding(words)
        case_embeds = self.case_embedding(cases)
        combined = torch.cat([word_embeds, case_embeds], dim=2)


        packed = nn.utils.rnn.pack_padded_sequence(
            combined,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed)

        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )

        x = self.linear(output)
        x = self.elu(x)
        logits = self.classifier(x)

        return logits

"""# **Create Task-2 Model**"""

vocab_size = len(word_to_index)
tagset_size = len(tag_to_index)
set_seed(42)
model_task2 = BiLSTM_Task2(
    vocab_size=len(word_to_index),
    tagset_size=len(tag_to_index),
    word_embedding_dim=100,
    case_embedding_dim=10,
    hidden_dim=256,
    pad_index=word_to_index["<PAD>"],
    num_case_types=5
)

print(model_task2.word_embedding.num_embeddings)
print(model_task2.case_embedding.num_embeddings)

"""# **Training for Task-2**"""

def train_model_task2(model, train_loader, dev_loader, tag_to_index, epochs = 10, learning_rate = 0.1, device = "cpu"):
  # Step-1: Set model to correct device
  model = model.to(device)

  # Step-2: Choose a loss function
  loss_function = nn.CrossEntropyLoss(ignore_index = tag_to_index["<PAD>"])

  # Step-3: Choose an Optimizer
  optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

  # Step-4: Repeat training for epochs
  for epoch in range(epochs):

    model.train()
    total_loss = 0.0

    # Step-5: Loop over one batch at a time
    for batch_words, batch_cases, batch_tags, batch_lengths in train_loader:

      # Step-6: Move batch data to cpu/gpu
      batch_words = batch_words.to(device)
      batch_tags = batch_tags.to(device)
      batch_cases = batch_cases.to(device)
      batch_lengths = batch_lengths.to(device)

      # Step-7: Clear all gradients from previous batch
      optimizer.zero_grad()

      # Step-8: Forward Pass
      logits = model(batch_words, batch_cases, batch_lengths)

      # Step-9: Flatten shapes so CrossEntropyLoss can read them
      logits = logits.reshape(-1, logits.shape[-1])
      true_data = batch_tags.reshape(-1)

      # Step-10: Compute how wrong the modle is
      loss = loss_function(logits, true_data)

      # Step-11: Backward Pass
      loss.backward()

      # Step-12: Update the model parameters
      optimizer.step()

      # Step-13: Total Loss
      total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

"""# **Create Devloader for Task-2**"""

# read dev data
dev_data = read_file("/content/dev")

# encode dev data with Task 2 encoder
dev_encoded_task2 = data_to_number_task2(dev_data, word_to_index, tag_to_index)

# wrap in Task 2 dataset
dev_dataset_task2 = NamedEntityRecognitionDataset_Task2(dev_encoded_task2)

# create Task 2 DataLoader
dev_loader_task2 = DataLoader(
    dev_dataset_task2,
    batch_size=32,
    shuffle=False,
    collate_fn=pad_batch_task2
)

"""# **Call Train Model Task-2 Function**"""

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)
train_model_task2(
    model_task2,
    train_loader_task2,
    dev_loader_task2,
    tag_to_index,
    epochs=15,
    learning_rate=0.7,
    device=device
)

"""# **Evaluation for Task-2**"""

def evaluate_model_task2(model, data_loader, device="cpu"):
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch_words, batch_cases, batch_tags, batch_lengths in data_loader:
            batch_words = batch_words.to(device)
            batch_cases = batch_cases.to(device)
            batch_tags = batch_tags.to(device)

            logits = model(batch_words, batch_cases, batch_lengths)

            # Take the highest-scoring tag for each token
            preds = torch.argmax(logits, dim=-1)

            # Go sentence by sentence
            for i in range(len(batch_lengths)):
                length = batch_lengths[i].item()

                # keep only the real tokens, not padding
                pred_seq = preds[i][:length].tolist()
                true_seq = batch_tags[i][:length].tolist()

                all_preds.append(pred_seq)
                all_true.append(true_seq)

    return all_preds, all_true

"""# **Read Glove and Build Embedding Matrix**"""

import numpy as np
import torch

def load_glove_embeddings(glove_path, word_to_index, embedding_dim=100):
    """
    create an embedding matrix where each row corresponds to a word id.
    If a word exists in Glove then use its pretrained vector.
    If a word does not exist in Glove then initialize it randomly.
    """

    # create a matrix of random values for all words first
    # shape: [vocab_size, embedding_dim]
    embedding_matrix = np.random.normal(
        loc=0.0,
        scale=0.1,
        size=(len(word_to_index), embedding_dim)
    )

    # make PAD token all zeros
    embedding_matrix[word_to_index["<PAD>"]] = np.zeros(embedding_dim)

    # read glove file line by line
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # skip empty lines if any
            if line == "":
                continue

            parts = line.split()

            # first item is the word
            word = parts[0]

            # rest are vector numbers
            vector = np.asarray(parts[1:], dtype=np.float32)

            # check if this word exists in our vocab
            if word in word_to_index:
                idx = word_to_index[word]
                embedding_matrix[idx] = vector

    # convert numpy array to torch tensor
    return torch.tensor(embedding_matrix, dtype=torch.float)

"""# **Load Matrix into the Model**"""

# wget http://nlp.stanford.edu/data/glove.6B.zip

# unzip glove.6B.zip

glove_path = "/content/glove.6B.100d.txt"

embedding_matrix = load_glove_embeddings(
                   glove_path = glove_path,
                   word_to_index = word_to_index,
                   embedding_dim = 100)
# copy pretrained vectors into the model's word embedding layer
model_task2.word_embedding.weight.data.copy_(embedding_matrix)

"""# **Freeze or Unfreeze Word Embeddings**"""

# model can adjust GloVe vectors during training
model_task2.word_embedding.weight.requires_grad = True

print(model_task2.word_embedding.weight.data.shape)
print(model_task2.word_embedding.weight.data[word_to_index["<PAD>"]][:5])

"""# **After loading Glove, train Task-2**"""

set_seed(42)
train_model_task2(
    model_task2,
    train_loader_task2,
    dev_loader_task2,
    tag_to_index,
    epochs = 20,
    learning_rate = 0.6,
    device = device
)

"""# **Evaluate Task-2 for Dev**"""

dev_prediction_task2, dev_true_task2 = evaluate_model_task2(
    model_task2,
    dev_loader_task2,
    device = device
)

"""precision, recall, f1 = compute_metrics(
    dev_prediction_task2,
    dev_true_task2,
    tag_to_index
)

print("Dev Precision: ", precision)
print("Dev Recall: ", recall)
print("Dev f1: ", f1) """

"""# **Write outputs in dev2.out**"""

write_predictions_to_file("dev2.out", dev_data, dev_prediction_task2, tag_to_index)

# python eval.py -g dev -p dev2.out

"""# **Create Task-2 TestLoader**"""

test_data = read_file("/content/test", has_tags = False)
test_sentences = []

for words, _ in test_data:
  dummy_tags = ["O"] * len(words)
  test_sentences.append((words, dummy_tags))

test_encoded_task2 = data_to_number_task2(
    test_sentences,
    word_to_index,
    tag_to_index
)

test_dataset_task2 = NamedEntityRecognitionDataset_Task2(test_encoded_task2)

test_loader_task2 = DataLoader(
    test_dataset_task2,
    batch_size = 32,
    shuffle = False,
    collate_fn = pad_batch_task2
)

"""# **Evaluate Task-2 for Test**"""

test_prediction_task2, _ = evaluate_model_task2(
    model_task2,
    test_loader_task2,
    device = device
)

# precision, recall, f1 = compute_metrics(
#     dev_prediction_task2,
#     dev_true_task2,
#     tag_to_index
# )

# print("Test Precision: ", precision)
# print("Test Recall: ", recall)
# print("Test f1: ", f1)

"""# **Write outputs in test2.out**"""

write_predictions_to_file("test2.out", test_data, test_prediction_task2, tag_to_index)

"""# **Task - 3: LSTM + CNN**"""

# build a character vocabulary
def build_char_vocab_task3(sentences):
    char_to_index = {"<PAD>": 0, "<UNK>": 1}
    next_char_id = 2

    for words, tags in sentences:
        for word in words:
            for ch in word:
                if ch not in char_to_index:
                    char_to_index[ch] = next_char_id
                    next_char_id += 1
    return char_to_index

# convert each word to character ids
def encode_data_task3(sentences, word_to_index, tag_to_index, char_to_index):

    encoded_sentences = []
    for words, tags in sentences:
        word_ids = []
        case_ids = []
        char_ids = []
        tag_ids = []

        for i in range(len(words)):
            word = words[i]
            tag = tags[i]

            # word id
            word_ids.append(word_to_index.get(word, word_to_index["<UNK>"]))

            # case id
            case_ids.append(get_case_id(word))

            # character ids for this word
            one_word_char_ids = []
            for ch in word:
                one_word_char_ids.append(char_to_index.get(ch, char_to_index["<UNK>"]))
            char_ids.append(one_word_char_ids)

            # tag id
            tag_ids.append(tag_to_index[tag])

        encoded_sentences.append((word_ids, case_ids, char_ids, tag_ids))
    return encoded_sentences

"""# **Prepare Dataset**"""

class NamedEntityRecognitionDataset_Task3(Dataset):
  def __init__(self, encoded_sentences):
        self.encoded_sentences = encoded_sentences

  def __len__(self):
        return len(self.encoded_sentences)

  def __getitem__(self, index):
        return self.encoded_sentences[index]

"""# **Encoded Training Data**"""

def pad_batch_task3(batch, pad_word_id=0, pad_case_id=0, pad_char_id=0, pad_tag_id=0):
    word_sequences = []
    case_sequences = []
    char_sequences = []
    tag_sequences = []
    lengths = []

    for words, cases, chars, tags in batch:
        word_sequences.append(words)
        case_sequences.append(cases)
        char_sequences.append(chars)
        tag_sequences.append(tags)
        lengths.append(len(words))

    max_len = max(lengths)

    max_word_len = 0
    for sent_chars in char_sequences:
        for word_chars in sent_chars:
            if len(word_chars) > max_word_len:
                max_word_len = len(word_chars)

    padded_words = []
    padded_cases = []
    padded_chars = []
    padded_tags = []

    for i in range(len(word_sequences)):
        words = word_sequences[i]
        cases = case_sequences[i]
        chars = char_sequences[i]
        tags = tag_sequences[i]

        # pad sentence level sequences
        word_padding = [pad_word_id] * (max_len - len(words))
        case_padding = [pad_case_id] * (max_len - len(cases))
        tag_padding = [pad_tag_id] * (max_len - len(tags))

        # pad char sequences for each word
        padded_char_sentence = []
        for word_chars in chars:
            char_padding = [pad_char_id] * (max_word_len - len(word_chars))
            padded_char_sentence.append(word_chars + char_padding)

        # pad fake words if sentence is shorter than max_len
        while len(padded_char_sentence) < max_len:
            padded_char_sentence.append([pad_char_id] * max_word_len)

        padded_words.append(words + word_padding)
        padded_cases.append(cases + case_padding)
        padded_chars.append(padded_char_sentence)
        padded_tags.append(tags + tag_padding)

    return (
        torch.tensor(padded_words, dtype=torch.long),
        torch.tensor(padded_cases, dtype=torch.long),
        torch.tensor(padded_chars, dtype=torch.long),
        torch.tensor(padded_tags, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )

"""# **Padding at Sentence Level and Word Level**"""

def pad_batch_task3(batch, pad_word_id=0, pad_case_id=0, pad_char_id=0, pad_tag_id=0):
    word_sequences = []
    case_sequences = []
    char_sequences = []
    tag_sequences = []
    lengths = []

    for words, cases, chars, tags in batch:
        word_sequences.append(words)
        case_sequences.append(cases)
        char_sequences.append(chars)
        tag_sequences.append(tags)
        lengths.append(len(words))

    max_len = max(lengths)

    # find the longest word length in the batch
    max_word_len = 0
    for sent_chars in char_sequences:
        for word_chars in sent_chars:
            if len(word_chars) > max_word_len:
                max_word_len = len(word_chars)

    padded_words = []
    padded_cases = []
    padded_chars = []
    padded_tags = []

    for i in range(len(word_sequences)):
        words = word_sequences[i]
        cases = case_sequences[i]
        chars = char_sequences[i]
        tags = tag_sequences[i]

        # pad word level sequences
        word_padding = [pad_word_id] * (max_len - len(words))
        case_padding = [pad_case_id] * (max_len - len(cases))
        tag_padding = [pad_tag_id] * (max_len - len(tags))

        # pad character sequences inside each word
        padded_char_sentence = []
        for word_chars in chars:
            char_padding = [pad_char_id] * (max_word_len - len(word_chars))
            padded_char_sentence.append(word_chars + char_padding)

        # if the sentence is shorter than max_len then pad fake words
        while len(padded_char_sentence) < max_len:
            padded_char_sentence.append([pad_char_id] * max_word_len)

        padded_words.append(words + word_padding)
        padded_cases.append(cases + case_padding)
        padded_chars.append(padded_char_sentence)
        padded_tags.append(tags + tag_padding)

    return (
        torch.tensor(padded_words, dtype=torch.long),
        torch.tensor(padded_cases, dtype=torch.long),
        torch.tensor(padded_chars, dtype=torch.long),
        torch.tensor(padded_tags, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long)
    )

"""# **Model: BiLSTM + CNN**"""

class BiLSTM_CNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        tagset_size,
        char_vocab_size,
        word_embedding_dim=100,
        case_embedding_dim=10,
        char_embedding_dim=30,
        char_cnn_out_dim=30,
        hidden_dim=256,
        pad_word_idx=0,
        pad_case_idx=0,
        pad_char_idx=0,
        num_case_types=5,
        kernel_size=3,
    ):
        super().__init__()

        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=word_embedding_dim,
            padding_idx=pad_word_idx,
        )

        self.case_embedding = nn.Embedding(
            num_embeddings=num_case_types,
            embedding_dim=case_embedding_dim,
            padding_idx=pad_case_idx,
        )

        self.char_embedding = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=char_embedding_dim,
            padding_idx=pad_char_idx,
        )

        self.char_cnn = nn.Conv1d(
            in_channels=char_embedding_dim,
            out_channels=char_cnn_out_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.lstm = nn.LSTM(
            input_size=word_embedding_dim + case_embedding_dim + char_cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.33,
        )

        self.linear = nn.Linear(hidden_dim * 2, 128)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(128, tagset_size)

    def forward(self, words, cases, chars, lengths):
        word_embeds = self.word_embedding(words)
        case_embeds = self.case_embedding(cases)

        B, S, W = chars.shape

        chars_flat = chars.reshape(B * S, W)
        char_embeds = self.char_embedding(chars_flat)
        char_embeds = char_embeds.transpose(1, 2)

        char_conv = self.char_cnn(char_embeds)
        char_conv = F.relu(char_conv)

        char_pool = torch.max(char_conv, dim=2).values
        char_features = char_pool.reshape(B, S, -1)

        combined = torch.cat([word_embeds, case_embeds, char_features], dim=2)

        packed = nn.utils.rnn.pack_padded_sequence(
            combined,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=words.size(1),
        )

        x = self.linear(output)
        x = self.elu(x)
        logits = self.classifier(x)
        return logits

"""# **Evaluation and Prediction**"""

def evaluate_model_task3(model, data_loader, device="cpu"):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch_words, batch_cases, batch_chars, batch_tags, batch_lengths in data_loader:
            batch_words = batch_words.to(device)
            batch_cases = batch_cases.to(device)
            batch_chars = batch_chars.to(device)
            batch_tags = batch_tags.to(device)

            logits = model(batch_words, batch_cases, batch_chars, batch_lengths)
            preds = torch.argmax(logits, dim=-1)

            for i in range(len(batch_lengths)):
                length = batch_lengths[i].item()
                all_preds.append(preds[i][:length].tolist())
                all_true.append(batch_tags[i][:length].tolist())

    return all_preds, all_true


def write_predictions_to_file(output_file, raw_sentences, predicted_seqs, tag_to_index):
    id_to_tag = {idx: tag for tag, idx in tag_to_index.items()}
    with open(output_file, "w") as f:
        for sent_idx in range(len(raw_sentences)):
            words, _ = raw_sentences[sent_idx]
            pred_seq = predicted_seqs[sent_idx]
            for i in range(len(words)):
                f.write(f"{i+1} {words[i]} {id_to_tag[pred_seq[i]]}\n")
            f.write("\n")

"""# **Build Data**"""

char_to_index = build_char_vocab_task3(train_data)

train_encoded_task3 = encode_data_task3(train_data, word_to_index, tag_to_index, char_to_index)
dev_encoded_task3 = encode_data_task3(dev_data, word_to_index, tag_to_index, char_to_index)

train_dataset_task3 = NamedEntityRecognitionDataset_Task3(train_encoded_task3)
dev_dataset_task3 = NamedEntityRecognitionDataset_Task3(dev_encoded_task3)

train_loader_task3 = DataLoader(
    train_dataset_task3,
    batch_size=32,
    shuffle=True,
    collate_fn=pad_batch_task3,
)

dev_loader_task3 = DataLoader(
    dev_dataset_task3,
    batch_size=32,
    shuffle=False,
    collate_fn=pad_batch_task3,
)

"""# **Load Glove into Word Embedding**"""

def load_glove_embeddings(glove_path, word_to_index, embedding_dim=100):
    embedding_matrix = np.random.normal(
        loc=0.0,
        scale=0.1,
        size=(len(word_to_index), embedding_dim),
    )
    embedding_matrix[word_to_index["<PAD>"]] = np.zeros(embedding_dim)

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            word = parts[0]
            vector = np.asarray(parts[1:], dtype=np.float32)
            if word in word_to_index:
                embedding_matrix[word_to_index[word]] = vector

    return torch.tensor(embedding_matrix, dtype=torch.float)

glove_path = "/content/glove.6B.100d.txt"
embedding_matrix = load_glove_embeddings(glove_path, word_to_index, embedding_dim=100)

"""# **Create and Train Model**"""

import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

model_task3 = BiLSTM_CNN(
    vocab_size=len(word_to_index),
    tagset_size=len(tag_to_index),
    char_vocab_size=len(char_to_index),
    word_embedding_dim=100,
    case_embedding_dim=10,
    char_embedding_dim=30,
    char_cnn_out_dim=30,
    hidden_dim=256,
    pad_word_idx=word_to_index["<PAD>"],
    pad_case_idx=0,
    pad_char_idx=char_to_index["<PAD>"],
    num_case_types=5,
).to(device)

model_task3.word_embedding.weight.data.copy_(embedding_matrix.to(device))
model_task3.word_embedding.weight.requires_grad = True


def train_model_task3(model, train_loader, tag_to_index, epochs=20, learning_rate=0.6, device="cpu"):
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=tag_to_index["<PAD>"])
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_words, batch_cases, batch_chars, batch_tags, batch_lengths in train_loader:
            batch_words = batch_words.to(device)
            batch_cases = batch_cases.to(device)
            batch_chars = batch_chars.to(device)
            batch_tags = batch_tags.to(device)

            optimizer.zero_grad()

            logits = model(batch_words, batch_cases, batch_chars, batch_lengths)
            logits = logits.reshape(-1, logits.shape[-1])
            true_data = batch_tags.reshape(-1)

            loss = loss_function(logits, true_data)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")


train_model_task3(
    model_task3,
    train_loader_task3,
    tag_to_index,
    epochs=20,
    learning_rate=0.6,
    device=device,
)

"""# **Dev Predictions**"""

dev_preds_task3, dev_gold_task3 = evaluate_model_task3(model_task3, dev_loader_task3, device=device)
write_predictions_to_file("dev3.out", dev_data, dev_preds_task3, tag_to_index)

# python eval.py -g dev -p dev3.out

"""# **Test Predictions**"""

test_data = read_file("/content/test", has_tags=False)

test_sentences_task3 = []
for words, _ in test_data:
    test_sentences_task3.append((words, ["O"] * len(words)))

test_encoded_task3 = encode_data_task3(test_sentences_task3, word_to_index, tag_to_index, char_to_index)
test_dataset_task3 = NamedEntityRecognitionDataset_Task3(test_encoded_task3)

test_loader_task3 = DataLoader(
    test_dataset_task3,
    batch_size=32,
    shuffle=False,
    collate_fn=pad_batch_task3,
)

test_preds_task3, _ = evaluate_model_task3(model_task3, test_loader_task3, device=device)
write_predictions_to_file("pred", test_data, test_preds_task3, tag_to_index)

"""# **Save all the task models**"""

# Task-1
torch.save(model.state_dict(), "blstm1.pt")

# Task-2
torch.save(model_task2.state_dict(), "blstm2.pt")

# Task-3
torch.save(model_task3.state_dict(), "blstm3.pt")

"""# **Readme.md**"""

readme_text = """# CSCI 544 - Homework 3
## Named Entity Recognition using BiLSTM Models

## 1. Overview
This assignment implements:
- Task 1: BiLSTM
- Task 2: BiLSTM + GloVe + Case Features
- Task 3: BiLSTM + CNN

## 2. Command Line Instructions

### Training
Run the notebook:
jupyter notebook HW3.ipynb

Open and run all cells.

Models saved as:
- blstm1.pt
- blstm2.pt
- blstm3.pt

### Prediction Files Generated
- dev1.out, test1.out
- dev2.out, test2.out
- dev3.out, pred

### Evaluation
- python eval.py -g dev -p dev1.out
- python eval.py -g dev -p dev2.out
- python eval.py -g dev -p dev3.out

## 3. Requirements
- train, dev, test files
- eval.py
- glove.6B.100d.txt
"""

with open("README.md", "w") as f:
    f.write(readme_text)