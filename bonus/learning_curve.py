import matplotlib.pyplot as plt
import sys
from pathlib import Path
from utils.analyser.network_loader import save_combined_network

class LearningCurve:
    def __init__(self, save_file=None, networks=None, stop_flag=None):
        """
        Initializes the learning curve tracker.

        Args:
            save_file (str): Path to save the networks in case of interruptions.
            networks (dict): The current networks being trained.
            stop_flag (threading.Event): Event flag for stopping training.
        """
        self.losses = []
        self.accuracies = []
        self.current_epoch = 0
        self.save_file = save_file
        self.networks = networks
        self.stop_flag = stop_flag

        # Initialize matplotlib interactive plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.loss_line, = self.ax.plot([], [], label='Loss', color='blue')
        self.accuracy_line, = self.ax.plot([], [], label='Accuracy', color='green')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')
        self.ax.legend()
        self.ax.set_title('Learning Curve')
        self.fig.show()

    def update(self, loss, accuracy):
        """
        Updates the learning curve with the latest loss and accuracy values.

        Args:
            loss (float): Loss value for the current epoch.
            accuracy (float): Accuracy value for the current epoch.
        """
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
        Finalizes the plot after training is complete.
        """
        plt.ioff()  # Turn off interactive mode
        plt.show()

    @staticmethod
    def compare_curves(curves, labels):
        """
        Compares multiple learning curves on the same graph.

        Args:
            curves (list): List of (losses, accuracies) tuples for each model.
            labels (list): List of labels for the corresponding models.
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
