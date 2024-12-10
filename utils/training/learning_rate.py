import numpy as np

def adjust_learning_rate(learning_rate, epoch, strategy="reduce_on_plateau", **kwargs):
    """
    Adjusts the learning rate dynamically based on the selected strategy.

    Args:
        learning_rate (float): Current learning rate.
        epoch (int): Current training epoch.
        strategy (str): Learning rate adjustment strategy.
        kwargs: Additional parameters for specific strategies.

    Returns:
        float: Updated learning rate.
    """
    if strategy == "reduce_on_plateau":
        val_acc = kwargs.get("val_acc", 0)
        best_val_acc = kwargs.get("best_val_acc", 0)
        patience = kwargs.get("patience", 5)  # Wait longer before reducing
        cooldown = kwargs.get("cooldown", 3)
        decay_factor = kwargs.get("decay_factor", 0.5)
        min_lr = kwargs.get("min_lr", 1e-4)
        threshold = kwargs.get("threshold", 0.002)
        last_plateau_epoch = kwargs.get("last_plateau_epoch", -cooldown)

        if epoch < patience:
            return learning_rate, last_plateau_epoch
        if (epoch - last_plateau_epoch >= patience) and (best_val_acc - val_acc) <= threshold:
            new_lr = max(min_lr, learning_rate * decay_factor)
            print(f"Reducing learning rate to {new_lr:.6f} due to plateau.")
            return new_lr, epoch
        
        return learning_rate, last_plateau_epoch
    elif strategy == "cyclic_lr": # Cyclic Learning Rate
        max_lr = kwargs.get("max_lr", 0.01)
        min_lr = kwargs.get("min_lr", 0.001)
        step_size = kwargs.get("step_size", 5)

        cycle = np.floor(1 + epoch / (2 * step_size))
        x = np.abs(epoch / step_size - 2 * cycle + 1)
        new_lr = min_lr + (max_lr - min_lr) * max(0, (1 - x))
        return new_lr

    elif strategy == "cosine_annealing_lr": # Cosine Annealing Learning Rate
        min_lr = kwargs.get("min_lr", 0.001)
        max_lr = kwargs.get("max_lr", 0.01)
        total_epochs = kwargs.get("total_epochs", 20)

        new_lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * epoch / total_epochs))
        return new_lr

    else:
        raise ValueError(f"Unknown learning rate strategy: {strategy}")