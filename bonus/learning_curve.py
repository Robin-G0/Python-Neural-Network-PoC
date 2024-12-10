import matplotlib.pyplot as plt
import signal
import sys

class LearningCurve:
    def __init__(self, save_file=None, networks=None, stop_flag=None):
        """
        Initializes the learning curve tracker.
        """
        self.losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.current_epoch = 0
        self.save_file = save_file
        self.networks = networks
        self.stop_flag = stop_flag

        # Initialize plot
        plt.ion()
        self.fig, self.ax = plt.subplots()

        self.loss_line, = self.ax.plot([], [], label='Loss', color='red')
        self.train_accuracy_line, = self.ax.plot([], [], label='Train Accuracy', color='blue')
        self.val_accuracy_line, = self.ax.plot([], [], label='Validation Accuracy', color='green')

        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Metrics')
        self.ax.legend()
        self.ax.set_title('Learning Curve')

        signal.signal(signal.SIGINT, self.handle_interrupt)

    def update(self, loss, train_accuracy, val_accuracy):
        """
        Updates the learning curve with the latest metrics.
        """
        if self.stop_flag and self.stop_flag.is_set():
            return

        self.losses.append(loss)
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)
        self.current_epoch += 1

        # Synchronize metrics with the plot
        epochs = range(1, self.current_epoch + 1)
        self.loss_line.set_data(epochs, self.losses)
        self.train_accuracy_line.set_data(epochs, self.train_accuracies)
        self.val_accuracy_line.set_data(epochs, self.val_accuracies)

        self.ax.relim()
        self.ax.autoscale_view()

        plt.pause(0.01)
        self.fig.canvas.flush_events()

    def finalize(self):
        """
        Finalizes the plot and stops the background thread.
        """
        plt.ioff()
        plt.show()

    def handle_interrupt(self, signum, frame):
        """
        Handle training interruptions gracefully.
        """
        print("\nTraining interrupted. Stopping...", file=sys.stderr)
        if self.stop_flag:
            self.stop_flag.set()
        self.finalize()
        plt.close('all')
        sys.exit(84)
