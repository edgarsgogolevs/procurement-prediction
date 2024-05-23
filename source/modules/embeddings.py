from collections.abc import Sequence
import logging

import numpy as np
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf

lg = logging.getLogger("embeddings")

MODEL_NAME = "AiLab-IMCS-UL/lvbert"
MAX_SEQ_LEN = 60
BATCH_SIZE = 128

lg.info("pulling models...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = TFAutoModel.from_pretrained(MODEL_NAME, from_pt=True)

lg.info("Creating embedding model...")

input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN, ),
                                  name="input_ids",
                                  dtype="int32")
mask = tf.keras.layers.Input(shape=(MAX_SEQ_LEN, ),
                             name="attention_mask",
                             dtype="int32")
embeddings = bert(input_ids, attention_mask=mask)[0]
net = tf.keras.layers.GlobalAveragePooling1D(name="pooling")(embeddings)
embedding_model = tf.keras.Model(inputs=[input_ids, mask], outputs=net)


def generate_tokens(texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    lg.info("Tokenizing texts...")
    x_ids = np.zeros((len(texts), MAX_SEQ_LEN))
    x_masks = np.zeros((len(texts), MAX_SEQ_LEN))
    for i, seq in enumerate(texts):
        tokens = tokenizer.encode_plus(seq,
                                       max_length=MAX_SEQ_LEN,
                                       truncation=True,
                                       padding="max_length",
                                       add_special_tokens=True,
                                       return_token_type_ids=False,
                                       return_attention_mask=True,
                                       return_tensors="tf")
        x_ids[i, :] = tokens["input_ids"]
        x_masks[i, :] = tokens["attention_mask"]
    return x_ids, x_masks


def generate_embeddings_from_tokens(x_ids: np.ndarray,
                                    x_masks: np.ndarray) -> np.ndarray:
    lg.info("Embeds from tokenized")
    dataset = tf.data.Dataset.from_tensor_slices((x_ids, x_masks))
    dataset = dataset.map(_map_func)
    dataset = dataset.batch(BATCH_SIZE)
    return embedding_model.predict(dataset)


def generate_embeddings_from_texts(texts: Sequence[str]) -> np.ndarray:
    lg.info("Embeds from texts")
    return generate_embeddings_from_tokens(*generate_tokens(texts))


def _map_func(input_ids, masks):
    return {"input_ids": input_ids, "attention_mask": masks}
