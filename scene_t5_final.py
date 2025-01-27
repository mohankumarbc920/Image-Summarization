from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from tensorflow.keras.applications import EfficientNetB7
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import uuid
import os
import pickle
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore")

IMAGES_PATH = "/home/mbaleath/mohan/scene_conda/scene_summz/dataset_im2p/stanford_images"
CAPTIONS_FILE = "/home/mbaleath/mohan/scene_conda/scene_summz/dataset_im2p/stanford_df_rectified.csv"
IMAGE_SIZE = (224, 224)
SEQ_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 50
FEATURES_CAPTIONS_FILE = "/home/mbaleath/mohan/scene_conda/scene_summz/dataset_im2p/features_captions_cross.pkl"

def load_paragraph_data(filename, images_path):
    print("Loading data...")
    data = pd.read_csv(filename)
    train_data, val_data, test_data = [], [], []

    for _, row in data.iterrows():
        img_name = os.path.join(images_path, f"{str(row['Image_name'])}.jpg")
        caption = row['Paragraph'].strip()
        if row['train']:
            train_data.append((img_name, caption))
        elif row['val']:
            val_data.append((img_name, caption))
        elif row['test']:
            test_data.append((img_name, caption))

    print("Sample from training data:")
    for i in range(min(5, len(train_data))):
        print(f"Image Path: {train_data[i][0]} | Caption: {train_data[i][1]}")

    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")
    return train_data, val_data, test_data

train_data, val_data, test_data = load_paragraph_data(CAPTIONS_FILE, IMAGES_PATH)

def preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def create_encoder():
    print("Creating encoder...")
    base_model = EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    return base_model

cnn_model = create_encoder()
print("EfficientNetB7 trainable:", cnn_model.trainable)

print("Loading T5 model...")
tokenizer = AutoTokenizer.from_pretrained("t5-small")
t5_model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-small")

t5_embedding_dim = t5_model.config.d_model 


class ImageToTextCrossAttention(tf.keras.Model):
    def __init__(self, embed_dim):
        super().__init__()
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu", name="image_to_t5_dense")
        self.layernorm = tf.keras.layers.LayerNormalization(name="image_to_t5_layernorm")
        # Cross-attention layer
        self.cross_attention = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=embed_dim, name="cross_attention")

    def call(self, image_features, text_embeddings):
        image_features = self.dense(image_features) 
        image_features = self.layernorm(image_features)  
        attention_output = self.cross_attention(query=text_embeddings, key=image_features, value=image_features)
        return attention_output

# Initialize projection layer with T5 embedding dimension
projection_layer = ImageToTextCrossAttention(t5_embedding_dim)

def extract_features(img_path, cnn_model, tokenizer, caption):
    img = preprocess_image(img_path)
    feature_map = cnn_model(tf.expand_dims(img, axis=0))
    batch, h, w, c = feature_map.shape
    image_features = tf.reshape(feature_map, (batch, h * w, c)) 

    tokenized_caption = tokenizer(caption, return_tensors="tf", padding="max_length", truncation=True, max_length=SEQ_LENGTH)
    token_ids = tokenized_caption["input_ids"]  

    text_embeddings = t5_model.shared(token_ids)  

    projected_sequence = projection_layer(image_features, text_embeddings) 
    return projected_sequence  

def prepare_dataset(data, cnn_model, tokenizer, max_seq_length):
    print("Preparing dataset...")
    images, captions = zip(*data)
    features, tokenized_captions = [], []

    for idx, (img_path, caption) in enumerate(zip(images, captions)):
        if idx % 1000 == 0:
            print(f"Processing {idx}/{len(images)}...")
        img_features = extract_features(img_path, cnn_model, tokenizer, caption)
        features.append(img_features)

        tokenized_caption = tokenizer(caption, max_length=max_seq_length, padding="max_length", truncation=True)
        tokenized_captions.append(tokenized_caption["input_ids"])

    return np.array(features), np.array(tokenized_captions)

if os.path.exists(FEATURES_CAPTIONS_FILE):
    print("Loading precomputed features and captions...")
    with open(FEATURES_CAPTIONS_FILE, "rb") as f:
        data = pickle.load(f)
        train_features, train_captions = data["train_features"], data["train_captions"]
        val_features, val_captions = data["val_features"], data["val_captions"]
else:
    print("Computing features and captions...")
    train_features, train_captions = prepare_dataset(train_data, cnn_model, tokenizer, SEQ_LENGTH)
    val_features, val_captions = prepare_dataset(val_data, cnn_model, tokenizer, SEQ_LENGTH)

    print("Saving features and captions...")
    with open(FEATURES_CAPTIONS_FILE, "wb") as f:
        pickle.dump({
            "train_features": train_features,
            "train_captions": train_captions,
            "val_features": val_features,
            "val_captions": val_captions
        }, f)

print(f"Train features shape: {train_features.shape}")  
print(f"Train captions shape: {train_captions.shape}")

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=3e-5,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_features_tensor = tf.convert_to_tensor(train_features, dtype=tf.float32)
train_captions_tensor = tf.convert_to_tensor(train_captions, dtype=tf.int32)
val_features_tensor = tf.convert_to_tensor(val_features, dtype=tf.float32)
val_captions_tensor = tf.convert_to_tensor(val_captions, dtype=tf.int32)

def calculate_bleu(reference_caption, generated_caption):
    reference_tokens = reference_caption.split()
    generated_tokens = generated_caption.split()
    smooth_fn = SmoothingFunction().method4
    bleu_1 = sentence_bleu([reference_tokens], generated_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
    bleu_2 = sentence_bleu([reference_tokens], generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
    bleu_3 = sentence_bleu([reference_tokens], generated_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn)
    bleu_4 = sentence_bleu([reference_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
    return bleu_1, bleu_2, bleu_3, bleu_4

def calculate_validation_loss(val_features, val_captions, t5_model):
    val_loss = 0
    steps = len(val_features) // BATCH_SIZE
    for i in range(steps):
        batch_features = val_features[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_captions = val_captions[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        if len(batch_features.shape) == 4:  
            batch_features = tf.squeeze(batch_features, axis=1)  

        attention_mask = tf.ones((batch_features.shape[0], batch_features.shape[1]), dtype=tf.int32)

        outputs = t5_model(
            input_ids=None,
            inputs_embeds=batch_features, 
            attention_mask=attention_mask,
            labels=batch_captions,
            training=False
        )

        val_loss += outputs.loss.numpy()

    val_loss /= steps
    return val_loss

    import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(image_path, generated_caption, cnn_model, t5_model, tokenizer):
    img = preprocess_image(image_path)
    img_features = cnn_model(tf.expand_dims(img, axis=0))  
    img_features = tf.reshape(img_features, (1, -1, img_features.shape[-1])) 

    tokenized_caption = tokenizer(
        text=generated_caption,  
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=SEQ_LENGTH
    )
    token_ids = tokenized_caption["input_ids"] 

    decoder_attention = t5_model(
        input_ids=token_ids,
        inputs_embeds=img_features,
        output_attentions=True,  
    ).decoder_attentions  

    attention_scores = decoder_attention[-1][0].numpy()  
    attention_scores = attention_scores.mean(axis=1) 
    attention_scores = attention_scores[:img_features.shape[1]].reshape((7, 7)) 

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(tf.image.decode_jpeg(tf.io.read_file(image_path)).numpy())
    plt.axis("off")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(attention_scores, cmap="viridis")
    plt.axis("off")
    plt.title("Attention Map")

    plt.show()

def train_model(train_features, train_captions, val_features, val_captions, t5_model, tokenizer, optimizer, loss_fn, epochs, batch_size, patience=5):
    print("Training started...")
    t5_model.encoder.trainable = True 
    t5_model.decoder.trainable = True 

    best_val_loss = float("inf") 
    no_improve_epochs = 0  

    steps_per_epoch = len(train_features) // batch_size 

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        total_loss = 0.0  

        for step in range(steps_per_epoch):
            try:
                batch_features = train_features[step * batch_size:(step + 1) * batch_size]
                batch_captions = train_captions[step * batch_size:(step + 1) * batch_size]

                attention_mask = tf.ones((batch_features.shape[0], batch_features.shape[1]), dtype=tf.int32)

                if len(batch_features.shape) == 4:  
                    batch_features = tf.squeeze(batch_features, axis=1)  
                with tf.GradientTape() as tape:
                    outputs = t5_model(
                        input_ids=None,
                        inputs_embeds=batch_features, 
                        attention_mask=attention_mask,
                        labels=batch_captions,
                        training=True
                    )
                    loss = outputs.loss

                grads = tape.gradient(loss, t5_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, t5_model.trainable_variables))

                total_loss += loss.numpy().item()

                print(f"Step {step + 1}/{steps_per_epoch} Loss: {loss.numpy()}")

            except Exception as e:
                print(f"Error during step {step + 1} in epoch {epoch + 1}: {e}")
                continue  

        avg_train_loss = float(total_loss / steps_per_epoch)  
        print(f"Epoch {epoch + 1} Average Training Loss: {avg_train_loss:.4f}")
        val_loss = calculate_validation_loss(val_features, val_captions, t5_model)
        val_loss = float(val_loss)  
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0  
            print(f"Validation loss improved to {best_val_loss:.4f}.")
        else:
            no_improve_epochs += 1
            print(f"No improvement in validation loss for {no_improve_epochs} epoch(s).")

        if no_improve_epochs >= patience: 
            print(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
            break

    print("=== Completed Training ===")

train_model(
    train_features=train_features_tensor,
    train_captions=train_captions_tensor,
    val_features=val_features_tensor,
    val_captions=val_captions_tensor,
    t5_model=t5_model,
    tokenizer=tokenizer,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    patience=3
)

def generate_caption(image_path, cnn_model, t5_model, tokenizer, max_seq_length):
    dummy_caption = "This is a dummy caption for feature extraction."

    img_features = extract_features(image_path, cnn_model, tokenizer, dummy_caption)  

    outputs = t5_model.generate(
        inputs_embeds=img_features,
        max_length=max_seq_length,
        num_beams=5,
        temperature=0.7,
        top_p=0.9,
        early_stopping=True
    )

    generated_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_caption

def enhanced_beam_search(
    cnn_model, t5_model, tokenizer, image_path, beam_width=3, max_seq_length=SEQ_LENGTH, repetition_penalty=1.5
):
    print("\nRunning Enhanced Beam Search...")
    
    dummy_caption = "This is a dummy caption for feature extraction."

    img_features = extract_features(image_path, cnn_model, tokenizer, dummy_caption)  

    start_token = tokenizer.encode("<pad>")[:-1]
    beams = [(start_token, 0.0, [])] 

    for _ in range(max_seq_length):
        new_beams = []
        for sequence, score, used_tokens in beams:
            input_ids = tf.convert_to_tensor([sequence], dtype=tf.int32)
            predictions = t5_model(
                input_ids=None,
                inputs_embeds=img_features,
                decoder_input_ids=input_ids,
                training=False
            )
            logits = predictions.logits[0, -1, :]

            top_indices = tf.argsort(logits, direction="DESCENDING")[:beam_width]
            top_scores = tf.nn.softmax(tf.gather(logits, top_indices))

            for index, s in zip(top_indices.numpy(), top_scores.numpy()):
                penalty = repetition_penalty if index in used_tokens else 0.0
                new_score = score + tf.math.log(s) - penalty
                new_sequence = sequence + [index]
                updated_tokens = used_tokens + [index]
                new_beams.append((new_sequence, new_score, updated_tokens))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

    best_sequence = beams[0][0]
    best_caption = tokenizer.decode(best_sequence, skip_special_tokens=True)
    return best_caption.strip()

def calculate_metrics(reference_caption, generated_caption):
    reference_tokens = reference_caption.split()
    generated_tokens = generated_caption.split()
    reference_set = set(reference_tokens)
    generated_set = set(generated_tokens)

    true_positives = len(reference_set & generated_set)
    false_positives = len(generated_set - reference_set)
    false_negatives = len(reference_set - generated_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives+false_positives)>0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives+false_negatives)>0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision+recall)>0 else 0
    accuracy = true_positives / len(reference_tokens) if len(reference_tokens)>0 else 0
    return precision, recall, f1, accuracy


def generate_and_evaluate(test_data, cnn_model, t5_model, tokenizer, sample_size=5, beam_width=3, repetition_penalty=1.5):
    sampled_data = random.sample(test_data, sample_size)
    total_bleu_1, total_bleu_2, total_bleu_3, total_bleu_4 = 0, 0, 0, 0
    total_precision, total_recall, total_f1, total_accuracy = 0, 0, 0, 0

    for image_path, reference_caption in sampled_data:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3).numpy()

        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Reference Caption:\n{reference_caption}", fontsize=10, wrap=True)
        ref_img_filename = f"reference_caption_image_{uuid.uuid4().hex}.png"
        plt.savefig(ref_img_filename)
        plt.close()
        print(f"Image with reference caption saved as: {ref_img_filename}")

        generated_caption = enhanced_beam_search(
            cnn_model, t5_model, tokenizer, image_path, beam_width=beam_width, repetition_penalty=repetition_penalty
        )

        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Generated Caption:\n{generated_caption}", fontsize=10, wrap=True)
        gen_img_filename = f"generated_caption_image_{uuid.uuid4().hex}.png"
        plt.savefig(gen_img_filename)
        plt.close()
        print(f"Image with generated caption saved as: {gen_img_filename}")

        bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu(reference_caption, generated_caption)
        print(f"BLEU-1: {bleu_1:.4f}, BLEU-2: {bleu_2:.4f}, BLEU-3: {bleu_3:.4f}, BLEU-4: {bleu_4:.4f}")
        total_bleu_1 += bleu_1
        total_bleu_2 += bleu_2
        total_bleu_3 += bleu_3
        total_bleu_4 += bleu_4

        precision, recall, f1, accuracy = calculate_metrics(reference_caption, generated_caption)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_accuracy += accuracy

    avg_bleu_1 = total_bleu_1 / sample_size
    avg_bleu_2 = total_bleu_2 / sample_size
    avg_bleu_3 = total_bleu_3 / sample_size
    avg_bleu_4 = total_bleu_4 / sample_size
    avg_precision = total_precision / sample_size
    avg_recall = total_recall / sample_size
    avg_f1 = total_f1 / sample_size
    avg_accuracy = total_accuracy / sample_size

    print(f"\nAverage BLEU-1 Score: {avg_bleu_1:.4f}")
    print(f"Average BLEU-2 Score: {avg_bleu_2:.4f}")
    print(f"Average BLEU-3 Score: {avg_bleu_3:.4f}")
    print(f"Average BLEU-4 Score: {avg_bleu_4:.4f}")
    #print(f"Average Precision: {avg_precision:.4f}, Average Recall: {avg_recall:.4f}, "
          #f"Average F1-Score: {avg_f1:.4f}, Average Accuracy: {avg_accuracy:.4f}")

generate_and_evaluate(test_data, cnn_model, t5_model, tokenizer, sample_size=5, beam_width=5, repetition_penalty=1.5)

image_path = test_data[0][0]  # Replace with an actual test image path
generated_caption = generate_and_evaluate(test_data, cnn_model, t5_model, tokenizer, sample_size=1, beam_width=5, repetition_penalty=1.5)
visualize_attention(image_path, generated_caption, cnn_model, t5_model, tokenizer)

print("==================================================== SAMPLE SIZE OF 1 ===============================================================")
print("")
print("==== one sample size with penalty = 1.5 =====")
generate_and_evaluate(test_data, cnn_model, t5_model, tokenizer, sample_size=1, beam_width=5, repetition_penalty=1.5)

print("==== one sample size with penalty = 1.5 =====")
generate_and_evaluate(test_data, cnn_model, t5_model, tokenizer, sample_size=1, beam_width=7, repetition_penalty=1.5)

print("==== one sample size with penalty = 1.8 ===== =========== TODAY ===============" )
generate_and_evaluate(test_data, cnn_model, t5_model, tokenizer, sample_size=1, beam_width=4, repetition_penalty=1.8)

print("==== one sample size with penalty = 2.0 =====")
generate_and_evaluate(test_data, cnn_model, t5_model, tokenizer, sample_size=1, beam_width=7, repetition_penalty=2.0)