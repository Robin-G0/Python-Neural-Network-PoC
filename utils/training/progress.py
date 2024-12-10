import sys

def print_progress_bar(current, total, length=40, prefix='', suffix='', fill='█', print_end="\r"):
    """
    Print a progress bar to the console.
    """
    percent = ("{0:.1f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end, file=sys.stderr)
    if current == total:
        print("", file=sys.stderr)