import re
import csv
import argparse
import sys
from collections import OrderedDict


class OscarLogParser:
    def __init__(self, select_epoch):
        super(OscarLogParser, self).__init__()
        self.select_epoch = select_epoch

    def parse(self, path):
        print(f"Parse {path}")
        train_time_regex = re.compile(".*Train Time: (?P<train_time>[0-9]*\.[0-9]*).*")
        global_step_regex = re.compile(
            ".*Epoch: (?P<epoch>[0-9]*), global_step: (?P<global_step>[0-9]*).*"
        )
        global_batch_size_regex = re.compile(
            ".*Total train batch size.* = (?P<global_batch_size>[0-9]*).*"
        )

        train_times = []
        global_steps = []
        global_batch_size = None

        with open(file=path, mode="r", encoding="UTF-8") as f:
            for line in f:
                match = train_time_regex.match(line)
                if match:
                    train_time = float(match["train_time"])
                    train_times.append(train_time)

                match = global_step_regex.match(line)
                if match:
                    global_step = float(match["global_step"])
                    global_steps.append(global_step)

                match = global_batch_size_regex.match(line)
                if match:
                    global_batch_size = int(match["global_batch_size"])

        metrics = OrderedDict()
        if self.select_epoch >= len(global_steps):
            print(
                f" the bigest epoch is {len(global_steps) - 1}, your select epoch is {self.select_epoch}, please select a smaller epoch."
            )
            sys.exit(0)
        if len(global_steps) > 0 and len(train_times) > 0:
            metrics["performance_step"] = self.select_epoch
            metrics["performance(it/s)"] = global_steps[self.select_epoch] / sum(
                train_times[: self.select_epoch + 1]
            )
            metrics["performance(samples/s)"] = metrics["performance(it/s)"] * global_batch_size

        print(metrics)
        return metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logpath",
        type=str,
        default="oscar.log",
        help="log path",
    )
    parser.add_argument("--epochs", type=int, default=0, help="0")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    parser = OscarLogParser(select_epoch=args.epochs)
    logfile = args.logpath

    print(f"logfile : {logfile}")
    metric = parser.parse(logfile)

    labels = list(metric.keys())
    try:
        with open(f"metrics.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=labels)
            writer.writeheader()
            writer.writerow(metric)
    except IOError:
        print("I/O error")
