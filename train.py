import os
import gc
import pandas as pd
import numpy as np
import pickle
from datasets import Dataset, DatasetDict
from datasets import Sequence, Value
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback
from transformers import TrainerCallback
import evaluate
import argparse
from functools import partial
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    hamming_loss, jaccard_score, accuracy_score
)

import warnings
warnings.filterwarnings('ignore')

def create_datasets_from_arrays(X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
    """
    Convert arrays into HuggingFace datasets format with specified structure
    
    Returns:
        DatasetDict with features:
        - dataset["train"]["text"]: text data
        - dataset["train"]["labels"]: multi-label arrays
        - dataset["val"]["text"]: validation text data (if provided)
        - dataset["val"]["labels"]: validation labels (if provided)
        - dataset["test"]["text"]: test text data (if provided)
        - dataset["test"]["labels"]: test labels (if provided)
    """
    # Create training dataset
    train_dict = {
        "text": X_train.tolist() if hasattr(X_train, 'tolist') else list(X_train),
        "labels": y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train)
    }
    
    datasets_dict = {
        "train": Dataset.from_dict(train_dict)
    }
    
    # Add validation dataset if provided
    if X_val is not None and y_val is not None:
        val_dict = {
            "text": X_val.tolist() if hasattr(X_val, 'tolist') else list(X_val),
            "labels": y_val.tolist() if hasattr(y_val, 'tolist') else list(y_val)
        }
        datasets_dict["val"] = Dataset.from_dict(val_dict)
    
    # Add test dataset if provided
    if X_test is not None and y_test is not None:
        test_dict = {
            "text": X_test.tolist() if hasattr(X_test, 'tolist') else list(X_test),
            "labels": y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
        }
        datasets_dict["test"] = Dataset.from_dict(test_dict)

    # Create DatasetDict
    dataset = DatasetDict(datasets_dict)
    
    return dataset

def preprocess_function(examples,tokenizer,max_length):
    """
    Proper tokenization function for multi-label classification.
    Ensures all outputs are compatible with HuggingFace Trainer.
    """
    # Handle batch vs single example
    if isinstance(examples['text'], str):
        texts = [examples['text']]
        labels = [examples['labels']]
    else:
        texts = examples['text']
        labels = examples['labels']
    
    # Tokenize the texts
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,  # Will be handled by data collator
        max_length=max_length,  # Adjust based on your model's limit
        return_tensors=None  # Don't return tensors yet, let data collator handle it
    )
    
    # Ensure labels are float32 for BCEWithLogitsLoss
    if isinstance(labels[0], (list, np.ndarray)):
        tokenized['labels'] = [np.array(label, dtype=np.float32).tolist() for label in labels]
    else:
        tokenized['labels'] = [np.array(labels, dtype=np.float32).tolist()]
    
    return tokenized

def sigmoid(x):
    """Sigmoid activation function"""
    return 1/(1 + np.exp(-x))

def comprehensive_evaluation(y_true, y_pred_proba, y_pred_binary=None, threshold=0.5):
    """
    Comprehensive evaluation for multi-label classification with all averaging methods
    
    Args:
        y_true: Ground truth binary labels (n_samples, n_labels)
        y_pred_proba: Predicted probabilities (n_samples, n_labels)
        y_pred_binary: Predicted binary labels (n_samples, n_labels), optional
        threshold: Threshold for converting probabilities to binary (default: 0.5)
    
    Returns:
        dict: Comprehensive metrics including all averaging methods
    """
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    metrics = {}
    
    try:
        # SAMPLES AVERAGE (per-sample then average across samples)
        metrics['precision_samples'] = precision_score(y_true, y_pred_binary, average='samples', zero_division=0)
        metrics['recall_samples'] = recall_score(y_true, y_pred_binary, average='samples', zero_division=0)
        metrics['f1_samples'] = f1_score(y_true, y_pred_binary, average='samples', zero_division=0)
        
        # MICRO AVERAGE (global aggregation)
        metrics['precision_micro'] = precision_score(y_true, y_pred_binary, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred_binary, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
        
        # MACRO AVERAGE (unweighted average across labels)
        metrics['precision_macro'] = precision_score(y_true, y_pred_binary, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred_binary, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
        
        # WEIGHTED AVERAGE (weighted by support)
        metrics['precision_weighted'] = precision_score(y_true, y_pred_binary, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred_binary, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred_binary, average='weighted', zero_division=0)
        
        # ACCURACY METRICS
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred_binary)
        
        # JACCARD (IoU) METRICS 
        metrics['jaccard_samples'] = jaccard_score(y_true, y_pred_binary, average='samples', zero_division=0)
        metrics['jaccard_macro'] = jaccard_score(y_true, y_pred_binary, average='macro', zero_division=0)
        metrics['jaccard_weighted'] = jaccard_score(y_true, y_pred_binary, average='weighted', zero_division=0)
        
        # ROC-AUC METRICS (using probabilities)
        try:
            metrics['roc_auc_micro'] = roc_auc_score(y_true, y_pred_proba, average='micro')
            metrics['roc_auc_macro'] = roc_auc_score(y_true, y_pred_proba, average='macro')
            metrics['roc_auc_weighted'] = roc_auc_score(y_true, y_pred_proba, average='weighted')
            metrics['roc_auc_samples'] = roc_auc_score(y_true, y_pred_proba, average='samples')
        except ValueError as e:
            print(f"Warning: ROC-AUC calculation failed: {e}")
            metrics['roc_auc_micro'] = 0.0
            metrics['roc_auc_macro'] = 0.0
            metrics['roc_auc_weighted'] = 0.0
            metrics['roc_auc_samples'] = 0.0
        
        # PR-AUC METRICS (using probabilities)
        try:
            metrics['pr_auc_micro'] = average_precision_score(y_true, y_pred_proba, average='micro')
            metrics['pr_auc_macro'] = average_precision_score(y_true, y_pred_proba, average='macro')
            metrics['pr_auc_weighted'] = average_precision_score(y_true, y_pred_proba, average='weighted')
            metrics['pr_auc_samples'] = average_precision_score(y_true, y_pred_proba, average='samples')
        except ValueError as e:
            print(f"Warning: PR-AUC calculation failed: {e}")
            metrics['pr_auc_micro'] = 0.0
            metrics['pr_auc_macro'] = 0.0
            metrics['pr_auc_weighted'] = 0.0
            metrics['pr_auc_samples'] = 0.0
        
    except Exception as e:
        print(f"Error in comprehensive_evaluation: {e}")
        # Return minimal metrics if calculation fails
        metrics = {
            'precision_micro': 0.0, 'recall_micro': 0.0, 'f1_micro': 0.0,
            'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0,
            'accuracy': 0.0, 'hamming_loss': 1.0
        }
    
    return metrics

def compute_metrics(eval_pred):
    """
    Enhanced compute_metrics function for transformers Trainer using comprehensive evaluation
    """
    predictions, labels = eval_pred
    
    # Apply sigmoid to get probabilities
    predictions_proba = sigmoid(predictions)
    
    # Convert to binary predictions using threshold 0.5
    predictions_binary = (predictions_proba > 0.5).astype(int)
    
    # Ensure labels are integers
    labels = labels.astype(int)
    
    # Use comprehensive evaluation
    metrics = comprehensive_evaluation(
        y_true=labels,
        y_pred_proba=predictions_proba,
        y_pred_binary=predictions_binary,
        threshold=0.5
    )
    
    # Return metrics with eval_ prefix for Trainer compatibility
    return {
        # Primary metrics for monitoring
        'eval_f1_micro': metrics['f1_micro'],
        'eval_f1_macro': metrics['f1_macro'],
        'eval_accuracy': metrics['accuracy'],
        'eval_hamming_loss': metrics['hamming_loss'],
        
        # Precision metrics
        'eval_precision_micro': metrics['precision_micro'],
        'eval_precision_macro': metrics['precision_macro'],
        'eval_precision_samples': metrics['precision_samples'],
        'eval_precision_weighted': metrics['precision_weighted'],
        
        # Recall metrics
        'eval_recall_micro': metrics['recall_micro'],
        'eval_recall_macro': metrics['recall_macro'],
        'eval_recall_samples': metrics['recall_samples'],
        'eval_recall_weighted': metrics['recall_weighted'],
        
        # F1 metrics
        'eval_f1_samples': metrics['f1_samples'],
        'eval_f1_weighted': metrics['f1_weighted'],
        
        # ROC-AUC metrics
        'eval_roc_auc_micro': metrics['roc_auc_micro'],
        'eval_roc_auc_macro': metrics['roc_auc_macro'],
        'eval_roc_auc_weighted': metrics['roc_auc_weighted'],
        'eval_roc_auc_samples': metrics['roc_auc_samples'],
        
        # PR-AUC metrics
        'eval_pr_auc_micro': metrics['pr_auc_micro'],
        'eval_pr_auc_macro': metrics['pr_auc_macro'],
        'eval_pr_auc_weighted': metrics['pr_auc_weighted'],
        'eval_pr_auc_samples': metrics['pr_auc_samples'],
        
        # Jaccard metrics
        'eval_jaccard_samples': metrics['jaccard_samples'],
        'eval_jaccard_macro': metrics['jaccard_macro'],
        'eval_jaccard_weighted': metrics['jaccard_weighted'],
    }

class ClearCUDACacheCallback(TrainerCallback):
    def on_step_end(self,args,state,control,**kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    def on_evaluate(self,args,state,control,**kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self,patience=1):
        super().__init__()
        self.patience=patience
        self.best_loss = float("inf")
        self.early_stop_count = 0

    def on_evaluate(self, args, state, control, **kwargs):
        # Access the evaluation loss
        eval_loss = kwargs["metrics"]["eval_loss"]
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.early_stop_count = 0
        else:
            self.early_stop_count += 1
        if self.early_stop_count >= self.patience:
            print("Early stopping triggered")
            control.should_training_stop = True
            
class LoggingCallback(TrainerCallback):
    def __init__(self,log_file):
        super().__init__()
        self.log_file=log_file
        self.last_train_loss = None
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}

        # Extract metrics
        epoch = round(float(state.epoch), 2) if state.epoch is not None else None
        train_loss = logs.get("loss")
        if train_loss is not None:
            train_loss = round(float(train_loss), 4)
            self.last_train_loss = train_loss
        elif self.last_train_loss is not None:
            train_loss = self.last_train_loss
        else:
            train_loss = "N/A"  # or skip logging this time
        eval_loss = logs.get("eval_loss")
        eval_accuracy = logs.get("eval_accuracy")
        eval_hamming_loss = logs.get("eval_hamming_loss")
        eval_jaccard_weighted = logs.get("eval_jaccard_weighted")

        # Round losses to 4 decimal places if present
        # train_loss = round(float(train_loss), 4) if train_loss is not None else None
        eval_loss = round(float(eval_loss), 4) if eval_loss is not None else None
        eval_accuracy = round(float(eval_accuracy), 4) if eval_accuracy is not None else None
        eval_hamming_loss = round(float(eval_hamming_loss), 4) if eval_hamming_loss is not None else None
        eval_jaccard_weighted = round(float(eval_jaccard_weighted), 4) if eval_jaccard_weighted is not None else None

        # Prepare log line: epoch, train_loss, eval_loss, eval_accuracy, eval_hamming_loss, eval_jaccard_weighted
        log_line = (
            f"Epoch: {epoch} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {eval_loss} | "
            f"Val Acc: {eval_accuracy} | "
            f"Val Hamming Loss: {eval_hamming_loss} | "
            f"Val Jaccard Weighted: {eval_jaccard_weighted}\n"
        )

        # Write to log file
        with open(self.log_file, "a") as f:
            f.write(log_line)
    
def main(args):
    dataset = create_datasets_from_arrays(X_train, y_train, X_val, y_val, X_test, y_test)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("{:<25}{:<15,}".format("Maximal context length:",tokenizer.model_max_length))
    print("{:<25}{:<15,}".format("Vocabulary size :",tokenizer.vocab_size))

    # Apply the tokenization function

    encode_function = partial(preprocess_function,tokenizer=tokenizer, max_length=args.max_context_length)
    
    tokenized_dataset = dataset.map(
        encode_function,
        batched=True,
        remove_columns=['text'],  # Remove the problematic text column
        desc="Tokenizing dataset"
    )
    
    # Define the proper feature type for multi-label classification
    label_feature = Sequence(Value("float32"), length=len(class_name))
    
    # Cast the labels column to float32 for all splits
    for split_name in tokenized_dataset.keys():
        tokenized_dataset[split_name] = tokenized_dataset[split_name].cast_column("labels", 
                                                                                  label_feature)
    print(f"Features: {list(tokenized_dataset['train'].features.keys())}")


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    class2id = {class_:id for id, class_ in enumerate(class_name)}
    id2class = {id:class_ for class_, id in class2id.items()}
    
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, 
                                                               num_labels=len(class_name),
                                                               id2label=id2class, 
                                                               label2id=class2id,
                                                               problem_type = "multi_label_classification"
                                                              )
    
    
    print()
    
    # Verify model is properly configured for multi-label classification
    print("🤖 Model Configuration Verification:")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Number of labels: {model.config.num_labels}")
    print(f"  Expected labels: {len(class_name)}")
    
    # Check if model configuration matches our data
    if model.config.num_labels != len(class_name):
        print(f"⚠️ WARNING: Model expects {model.config.num_labels} labels, but data has {len(class_name)}")
        print("  This might cause issues during training")
    else:
        print(f"✅ Model configuration matches data: {len(class_name)} labels")
    
    # Verify model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Fix tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    training_args = TrainingArguments(
        # Output and logging
        output_dir=args.output_dir,
        # logging_dir="./logs",
        # logging_steps=100,
        logging_strategy="no", ## we customize logging intead of using the built-in logging
        
        # Learning parameters
        learning_rate=2e-5,
        lr_scheduler_type="linear",  # Linear decay
        warmup_ratio=0.1,  # 10% warmup
        weight_decay=0.01,
        
        # Batch sizes (adjust based on GPU memory)
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.gradient_accumulation_step,  # Effective batch size = 4 * 12 = 48

        # Training epochs and evaluation
        num_train_epochs=args.num_epochs,  # Increased for better convergence
        eval_strategy="steps",  # More frequent evaluation
        eval_steps=100,  # Evaluate every 100 steps
        
        # 🎯 OPTIMAL METRICS FOR MULTI-LABEL CLASSIFICATION
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,  # Keep only 3 best checkpoints
        load_best_model_at_end=True,
        
        # 🔥 RECOMMENDED: Use Hamming Loss for multi-label problems
        metric_for_best_model="eval_hamming_loss",  # Primary metric: lower is better
        greater_is_better=False,  # Hamming loss: lower = better performance
        
        # Alternative good options:
        # metric_for_best_model="eval_f1_micro",     # Current choice - also excellent
        # metric_for_best_model="eval_jaccard_samples", # IoU metric - good for multi-label
        
        # Memory and performance optimization
        dataloader_pin_memory=False,  # Disable to avoid forking issues
        dataloader_num_workers=0,     # Disable multiprocessing
        remove_unused_columns=False,  # Keep all columns for multi-label
        
        # Mixed precision for faster training (if GPU supports it)
        fp16=True,  # Enable if using compatible GPU
        
        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,
        
        # Report metrics
        report_to=None,  # Disable wandb/tensorboard if not needed
        run_name="multi_label_posture_classification",
    )
    # # Early stopping callback for overfitting control
    # early_stopping = EarlyStoppingCallback(
    #     early_stopping_patience=3,  # Stop if no improvement for 3 evaluations
    #     early_stopping_threshold=0.001  # Minimum improvement threshold
    # )
    
    # Initialize trainer with enhanced configuration (using processing_class)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        processing_class=tokenizer,  # Updated parameter name
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(patience=3),
                   ClearCUDACacheCallback(),
                   LoggingCallback(log_file=os.path.join(args.output_dir,"training_logs.txt"))],  # Add early stopping callback
    )    

    print("\n🎯 Starting training...")
    trainer.train()
    print("✅ Training completed successfully!")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Fine-tune MordenBERT')

    parser.add_argument("--seed",  type=int,default=42)
    parser.add_argument("--data_path", type=str, default='processed_data')
    parser.add_argument("--output_dir", type=str, default='model_output')
    parser.add_argument('--model_path', type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument('--train_batch', type=int, default=4)
    parser.add_argument('--eval_batch', type=int, default=8)
    parser.add_argument('--gradient_accumulation_step', type=int, default=12)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--max_context_length', type=int, default=8192)
    
    args= parser.parse_args()

    ### Load Dataset for model training and evaluation ###
    data_path=os.path.join(os.path.dirname(__file__), args.data_path)
    with open(os.path.join(data_path,'train_arrays.pkl'), 'rb') as f:
        train_data = pickle.load(f)
        X_train = train_data['X_train']
        y_train = train_data['y_train']
    
    with open(os.path.join(data_path,'val_arrays.pkl'), 'rb') as f:
        val_data = pickle.load(f)
        X_val = val_data['X_val']
        y_val = val_data['y_val']
    
    with open(os.path.join(data_path,'test_arrays.pkl'), 'rb') as f:
        test_data = pickle.load(f)
        X_test = test_data['X_test']
        y_test = test_data['y_test']
    
    with open(os.path.join(data_path,'class_name.pkl'), 'rb') as f:
        class_name_data = pickle.load(f)
        class_name = class_name_data['class_name']

    main(args)
    


    