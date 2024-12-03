import matplotlib.pyplot as plt
import numpy as np
import signal
import sys
from pathlib import Path
from utils.analyser.network_loader import save_network

class LearningCurve:
    def __init__(self):
        self.losses = []
        self.accuracies = []
        self.current_epoch = 0

        # Initialize the plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.loss_line, = self.ax.plot([], [], label='Loss', color='blue')
        self.accuracy_line, = self.ax.plot([], [], label='Accuracy', color='green')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Value')
        self.ax.legend()
        self.ax.set_title('Learning Curve')
        self.fig.show()

    def _handle_interrupt(self, signum, frame):
        """
        Handles CTRL + C (SIGINT) to save progress and stop training.
        """
        if self.stop_flag:  # Prevent multiple calls
            return

        print("\nTraining interrupted. Saving current progress...", file=sys.stderr)
        interrupted_save_file = str(Path(self.save_file).with_name(f"{Path(self.save_file).stem}_interrupted.nn"))
        save_network(self.network, interrupted_save_file)
        print(f"Progress saved to {interrupted_save_file}", file=sys.stderr)
        self.stop_flag = True

    def update(self, loss, accuracy):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.current_epoch += 1

        # Update plot data
        self.loss_line.set_data(range(1, self.current_epoch + 1), self.losses)
        self.accuracy_line.set_data(range(1, self.current_epoch + 1), self.accuracies)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.01)
        self.fig.canvas.flush_events()

    def finalize(self):
        """
        Finalizes the learning curve plot after training completes.
        """
        plt.ioff()  # Turn off interactive mode
        plt.show()

    @staticmethod
    def compare_curves(curves, labels):
        """
        Compares multiple learning curves on the same graph.
        Args:
            curves: List of (losses, accuracies) tuples for each model.
            labels: List of labels for the corresponding models.
        """
        plt.figure()
        for idx, (losses, accuracies) in enumerate(curves):
            plt.plot(range(1, len(losses) + 1), losses, label=f"{labels[idx]} Loss")
            plt.plot(range(1, len(accuracies) + 1), accuracies, label=f"{labels[idx]} Accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Comparison of Learning Curves')
        plt.legend()
        plt.show()
