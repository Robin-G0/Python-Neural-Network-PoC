import numpy as np
import sys
import copy
from numba import cuda
from utils.training.progress import print_progress_bar
from utils.training.learning_rate import adjust_learning_rate
from utils.training.training_utils import forward_pass, compute_loss, compute_gradients, update_weights

@cuda.jit
def apply_dropout_kernel(inputs, dropout_mask, dropout_rate):
    idx = cuda.grid(1)
    if idx < inputs.size:
        dropout_mask[idx] = 1 if np.random.rand() > dropout_rate else 0
        inputs[idx] *= dropout_mask[idx]

def apply_dropout(inputs, dropout_rate):
    if dropout_rate <= 0.0 or dropout_rate >= 1.0:
        return inputs  # No dropout if rate is invalid
    inputs_device = cuda.to_device(inputs)
    dropout_mask_device = cuda.device_array_like(inputs)
    threads_per_block = 256
    blocks_per_grid = (inputs.size + threads_per_block - 1) // threads_per_block
    apply_dropout_kernel[blocks_per_grid, threads_per_block](inputs_device, dropout_mask_device, dropout_rate)
    return inputs_device.copy_to_host()

def train_network_multithreaded(network, data, initial_learning_rate=0.001, epochs=20, batch_size=500, 
                                updates_queue=None, stop_flag=None, lr_strategy="reduce_on_plateau", lr_params=None, dropout_rate=0.2):
    if lr_params is None:
        lr_params = {}

    num_samples = len(data)
    total_batches = (num_samples + batch_size - 1) // batch_size

    # Split data into train/validation
    split_idx = int(num_samples * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    learning_rate = initial_learning_rate
    best_val_acc = 0
    last_plateau_epoch = -1
    best_epoch_loss = float('inf')
    best_epoch_loss_val = 1.0
    best_epoch = -1
    best_network_state = None
    consecutive_increase_count = 0

    for epoch in range(epochs):
        if stop_flag and stop_flag.is_set():
            print("Training stopped by user.", file=sys.stderr)
            break

        # Adjust Learning Rate
        if lr_strategy == "reduce_on_plateau":
            learning_rate, last_plateau_epoch = adjust_learning_rate(
                learning_rate, epoch, strategy=lr_strategy,
                val_acc=best_val_acc, last_plateau_epoch=last_plateau_epoch, **lr_params
            )
        else:
            learning_rate = adjust_learning_rate(
                learning_rate, epoch, strategy=lr_strategy, **lr_params
            )

        print(f"Epoch {epoch + 1}/{epochs} - Learning Rate: {learning_rate:.6f}", file=sys.stderr)
        np.random.shuffle(train_data)

        epoch_loss, correct_predictions = 0.0, 0

        for start_idx in range(0, len(train_data), batch_size):
            if stop_flag and stop_flag.is_set():
                print("Training stopped by user.", file=sys.stderr)
                break

            batch = list(train_data)[start_idx:start_idx + batch_size]
            batch_loss = 0.0

            for inputs, targets in batch:
                # Apply dropout only during training
                inputs_dropped = apply_dropout(inputs, dropout_rate)
                outputs = forward_pass(network, inputs_dropped)
                loss = compute_loss(outputs, targets)
                gradients = compute_gradients(network, outputs, targets, regularization=0.001)
                update_weights(network, gradients, learning_rate)

                batch_loss += loss
                correct_predictions += int((outputs.round() == targets).all())

            print_progress_bar(
                start_idx // batch_size + 1, total_batches, prefix='Progress:', suffix='Complete', length=20
            )
            epoch_loss += batch_loss / len(batch)

        # Evaluate validation accuracy
        val_outputs = np.array([forward_pass(network, inputs) for inputs, _ in val_data])
        val_targets = np.array([targets for _, targets in val_data])

        if val_outputs.ndim > 1 and val_outputs.shape[1] > 1:  # Multi-class classification
            val_predictions = np.argmax(val_outputs, axis=1)
            val_target_labels = np.argmax(val_targets, axis=1)
        else:  # Binary classification
            val_predictions = (val_outputs >= 0.5).astype(int).ravel()
            val_target_labels = val_targets.astype(int).ravel()

        val_correct = np.sum(val_predictions == val_target_labels)
        val_accuracy = val_correct / len(val_data)
        best_val_acc = max(best_val_acc, val_accuracy)

        if updates_queue:
            updates_queue.put((epoch_loss / total_batches, correct_predictions / len(train_data), val_accuracy))

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / total_batches:.4f} - "
              f"Train Accuracy: {correct_predictions / len(train_data):.4f} - Val Accuracy: {val_accuracy:.4f}",
              file=sys.stderr)

        # Check for early stopping
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            best_epoch = epoch
            best_network_state = copy.deepcopy(network)
            best_epoch_loss_val = epoch_loss / total_batches
            consecutive_increase_count = 0  # Reset the counter
        elif epoch_loss > best_epoch_loss:
            consecutive_increase_count += 1
            if consecutive_increase_count >= 3:
                print(f"Stopping early at epoch {epoch + 1} due to 3 consecutive increases in loss.", file=sys.stderr)
                break
        else:
            consecutive_increase_count = 0  # Reset the counter if loss does not increase

    # Restore the best network state
    if best_network_state:
        network.update(best_network_state)
        print(f"Restored network state from epoch {best_epoch + 1} with lowest loss {best_epoch_loss_val:.4f}.", file=sys.stderr)