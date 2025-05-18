import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras    
import pandas as pd
import seaborn as sns
from datetime import datetime
import json

# Custom callback for detailed training logs
class DetailedTensorBoard(keras.callbacks.Callback):
    def __init__(self, log_dir, histogram_freq=1, batch_size=32, update_freq='epoch'):
        super(DetailedTensorBoard, self).__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.histogram_freq = histogram_freq
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.epoch = 0  # Initialize epoch attribute
        
        # Training metrics to track
        self.metrics = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'lr': []
        }
        self.epoch_logs = []
        self.batch_logs = []
        
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1} starting...")
        self.epoch = epoch  # Update epoch at the beginning of each epoch
        self.batch_logs_current_epoch = []
        
    def on_batch_end(self, batch, logs=None):
        # Record detailed batch-level metrics
        if logs is not None and batch % 10 == 0:  # Log every 10 batches
            batch_log = {'epoch': self.epoch, 'batch': batch}
            batch_log.update({k: float(v) for k, v in logs.items()})
            self.batch_logs_current_epoch.append(batch_log)
            self.batch_logs.append(batch_log)
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        
        # Extract and store metrics from logs
        if logs is not None:
            # Record epoch metrics
            for metric in self.metrics:
                if metric in logs:
                    self.metrics[metric].append(logs[metric])
            
            # Store current learning rate
            try:
                current_lr = float(self.model.optimizer.lr)
                if callable(self.model.optimizer.lr):
                    current_lr = float(self.model.optimizer.lr(self.epoch).numpy())
                self.metrics['lr'].append(current_lr)
            except:
                self.metrics['lr'].append(None)
            
            # Save comprehensive epoch log
            epoch_log = {'epoch': epoch}
            epoch_log.update({k: float(v) for k, v in logs.items()})
            epoch_log['lr'] = self.metrics['lr'][-1]
            self.epoch_logs.append(epoch_log)
            
            # Print summary of epoch
            print(f"Epoch {epoch+1} completed:")
            for k, v in logs.items():
                print(f"  {k}: {v:.4f}")
            if self.metrics['lr'][-1] is not None:
                print(f"  Learning rate: {self.metrics['lr'][-1]:.6f}")
            
            # Save batch logs for current epoch
            if self.batch_logs_current_epoch:
                batch_log_file = os.path.join(self.log_dir, f"batch_logs_epoch_{epoch+1}.json")
                with open(batch_log_file, 'w') as f:
                    json.dump(self.batch_logs_current_epoch, f, indent=2)
        
        # Save epoch logs
        epoch_log_file = os.path.join(self.log_dir, "epoch_logs.json")
        with open(epoch_log_file, 'w') as f:
            json.dump(self.epoch_logs, f, indent=2)
        
        # Generate and save training progress visualization
        self._plot_training_progress()
    
    def on_train_end(self, logs=None):
        # Save final metrics
        metrics_file = os.path.join(self.log_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Final plots
        self._plot_training_progress()
        self._plot_batch_metrics()
        self._plot_learning_rate()
        
        print(f"\nTraining completed. Logs saved to {self.log_dir}")
    
    def _plot_training_progress(self):
        # Create comprehensive training progress visualization
        if len(self.metrics['loss']) > 0:
            plt.figure(figsize=(15, 10))
            
            # Plot loss
            plt.subplot(2, 2, 1)
            plt.plot(self.metrics['loss'], label='Training Loss')
            if 'val_loss' in self.metrics and len(self.metrics['val_loss']) > 0:
                plt.plot(self.metrics['val_loss'], label='Validation Loss')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot accuracy
            plt.subplot(2, 2, 2)
            plt.plot(self.metrics['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in self.metrics and len(self.metrics['val_accuracy']) > 0:
                plt.plot(self.metrics['val_accuracy'], label='Validation Accuracy')
            plt.title('Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot learning rate
            if 'lr' in self.metrics and len(self.metrics['lr']) > 0 and self.metrics['lr'][0] is not None:
                plt.subplot(2, 2, 3)
                plt.plot(self.metrics['lr'])
                plt.title('Learning Rate')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot validation metrics if available
            if 'val_loss' in self.metrics and len(self.metrics['val_loss']) > 0:
                plt.subplot(2, 2, 4)
                epochs = list(range(1, len(self.metrics['val_loss']) + 1))
                plt.scatter(epochs, self.metrics['val_loss'], label='Val Loss', alpha=0.7)
                plt.scatter(epochs, self.metrics['val_accuracy'], label='Val Acc', alpha=0.7)
                
                # Add trend lines only if we have enough data points (at least 3)
                if len(epochs) >= 3:
                    # Add trend lines
                    z1 = np.polyfit(epochs, self.metrics['val_loss'], 1)
                    p1 = np.poly1d(z1)
                    plt.plot(epochs, p1(epochs), "r--", alpha=0.5)
                    
                    z2 = np.polyfit(epochs, self.metrics['val_accuracy'], 1)
                    p2 = np.poly1d(z2)
                    plt.plot(epochs, p2(epochs), "g--", alpha=0.5)
                
                plt.title('Validation Metrics')
                plt.xlabel('Epoch')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'training_progress.png'), dpi=150)
            plt.close()
    
    def _plot_batch_metrics(self):
        # Plot batch-level metrics
        if len(self.batch_logs) > 0:
            df = pd.DataFrame(self.batch_logs)
            
            metrics_to_plot = [col for col in df.columns if col not in ['epoch', 'batch']]
            if metrics_to_plot:
                plt.figure(figsize=(15, 5 * len(metrics_to_plot)))
                
                for i, metric in enumerate(metrics_to_plot):
                    plt.subplot(len(metrics_to_plot), 1, i+1)
                    sns.lineplot(x='batch', y=metric, hue='epoch', data=df, palette='viridis')
                    plt.title(f'Batch {metric}')
                    plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.log_dir, 'batch_metrics.png'), dpi=150)
                plt.close()
    
    def _plot_learning_rate(self):
        # Plot detailed learning rate
        if 'lr' in self.metrics and len(self.metrics['lr']) > 0 and self.metrics['lr'][0] is not None:
            plt.figure(figsize=(10, 6))
            lr_values = self.metrics['lr']
            plt.plot(lr_values)
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')  # Logarithmic scale often better for LR visualization
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'learning_rate.png'), dpi=150)
            plt.close()

# Custom callback to visualize layer activations
class ActivationVisualizer(keras.callbacks.Callback):
    def __init__(self, validation_data, layer_names, log_dir, num_samples=3, freq=5):
        super(ActivationVisualizer, self).__init__()
        self.X_val, self.y_val = validation_data
        self.layer_names = layer_names  # List of layer names to visualize
        self.log_dir = os.path.join(log_dir, 'activations')
        os.makedirs(self.log_dir, exist_ok=True)
        self.num_samples = min(num_samples, len(self.X_val))
        self.freq = freq  # Visualize every N epochs
        self.sample_indices = np.random.choice(len(self.X_val), self.num_samples, replace=False)
    
    def on_epoch_end(self, epoch, logs=None):
        # Only visualize on specified frequency
        if (epoch + 1) % self.freq != 0:
            return
        
        # Create a new model that outputs layer activations
        layer_outputs = [layer.output for layer in self.model.layers if layer.name in self.layer_names]
        if not layer_outputs:
            print(f"Warning: None of the specified layer names {self.layer_names} were found in the model.")
            return
            
        activation_model = keras.models.Model(inputs=self.model.input, outputs=layer_outputs)
        
        # Create a directory for this epoch
        epoch_dir = os.path.join(self.log_dir, f'epoch_{epoch+1}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Get activations for sample images
        for i, idx in enumerate(self.sample_indices):
            sample_img = np.expand_dims(self.X_val[idx], axis=0)
            activations = activation_model.predict(sample_img)
            
            # Create visualization for each layer
            for j, (layer_name, layer_activation) in enumerate(zip(self.layer_names, activations)):
                # Get number of features in the layer
                if len(layer_activation.shape) == 4:  # Conv layers (batch, width, height, channels)
                    n_features = layer_activation.shape[-1]
                    width = layer_activation.shape[1]
                    height = layer_activation.shape[2]
                    
                    # Create figure with all feature maps
                    display_grid_size = int(np.ceil(np.sqrt(n_features)))
                    display_grid_width = display_grid_size
                    display_grid_height = display_grid_size
                    
                    # Limit the number of feature maps to 64 for visualization
                    n_features = min(n_features, 64)
                    
                    # Create figure
                    fig = plt.figure(figsize=(15, 15))
                    
                    for k in range(n_features):
                        ax = plt.subplot(display_grid_height, display_grid_width, k + 1)
                        feature_map = layer_activation[0, :, :, k]
                        
                        # Normalize for display
                        feature_map -= feature_map.mean()
                        if feature_map.std() > 0:
                            feature_map /= feature_map.std()
                        feature_map *= 64
                        feature_map += 128
                        feature_map = np.clip(feature_map, 0, 255).astype('uint8')
                        
                        plt.imshow(feature_map, cmap='viridis')
                        ax.set_xticks([])
                        ax.set_yticks([])
                    
                    # Save figure
                    plt.suptitle(f"Layer: {layer_name} - Sample {i+1}", fontsize=16)
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.95)
                    plt.savefig(os.path.join(epoch_dir, f'sample_{i+1}_layer_{layer_name}.png'), dpi=150)
                    plt.close(fig)
                    
                elif len(layer_activation.shape) == 2:  # Dense layers
                    # For dense layers, create a bar plot
                    fig = plt.figure(figsize=(10, 6))
                    plt.bar(range(layer_activation.shape[1]), layer_activation[0])
                    plt.title(f"Layer: {layer_name} - Sample {i+1}")
                    plt.xlabel('Neuron')
                    plt.ylabel('Activation')
                    plt.tight_layout()
                    plt.savefig(os.path.join(epoch_dir, f'sample_{i+1}_layer_{layer_name}.png'), dpi=150)
                    plt.close(fig)
        
        print(f"Saved activation visualizations for epoch {epoch+1}")

# Function to apply enhanced monitoring to model training
def setup_enhanced_monitoring(model_save_path='./output_files_final'):
    """
    Create enhanced monitoring callbacks for model training
    
    Args:
        model_save_path: Base directory for saving model and logs
        
    Returns:
        List of callbacks to add to model.fit()
    """
    # Create timestamp for this training run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(model_save_path, f'training_logs_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    # List of callbacks
    callbacks = []
    
    # Model checkpoint callback - save best model
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_save_path, 'best_model.keras'),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    callbacks.append(model_checkpoint)
    
    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,  # Increased patience
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Learning rate scheduler - cosine decay with warm restarts
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    callbacks.append(lr_scheduler)
    
    # Custom TensorBoard with enhanced logging
    tensorboard = DetailedTensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch'
    )
    callbacks.append(tensorboard)
    
    return callbacks, log_dir

def add_activation_visualizer(callbacks, log_dir, model, validation_data):
    """
    Add activation visualization to callbacks
    
    Args:
        callbacks: List of callbacks
        log_dir: Directory for saving visualizations
        model: The model being trained
        validation_data: Tuple of (X_val, y_val)
        
    Returns:
        Updated list of callbacks
    """
    # Find convolutional layers to visualize
    conv_layers = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D) or layer.name.startswith('conv2d'):
            conv_layers.append(layer.name)
    
    # Select a subset of layers based on depth
    selected_layers = []
    if len(conv_layers) >= 3:
        # Take first, middle, and last convolutional layers
        selected_layers.append(conv_layers[0])
        selected_layers.append(conv_layers[len(conv_layers)//2])
        selected_layers.append(conv_layers[-1])
    else:
        selected_layers = conv_layers
    
    # Add a dense layer if available
    dense_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Dense) and layer.name != model.layers[-1].name:
            dense_layer = layer.name
            break
    
    if dense_layer:
        selected_layers.append(dense_layer)
    
    # Add activation visualizer
    activation_viz = ActivationVisualizer(
        validation_data=validation_data,
        layer_names=selected_layers,
        log_dir=log_dir,
        num_samples=3,
        freq=5
    )
    callbacks.append(activation_viz)
    
    return callbacks

# Use these functions in your main training script:
"""
# Example usage in main-fruit-classification.py:

# Set up enhanced monitoring callbacks
callbacks, log_dir = setup_enhanced_monitoring(OUTPUT_DIR)

# Add activation visualizer after model is defined
callbacks = add_activation_visualizer(callbacks, log_dir, model, (X_val, y_val_one_hot))

# Use callbacks in model.fit
history = model.fit(
    X_train, y_train_one_hot,
    batch_size=64, epochs=epochs, 
    callbacks=callbacks,
    validation_data=(X_val, y_val_one_hot),
    class_weight=class_weight_dict
)
"""

print("Enhanced model monitoring module loaded. Use setup_enhanced_monitoring() to create callbacks.") 