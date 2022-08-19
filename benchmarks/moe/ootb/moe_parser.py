import re
import argparse
import csv
from collections import OrderedDict


class MoeLogParser():
    def __init__(self, select_epoch):
        super(MoeLogParser, self).__init__()
        self.select_epoch = select_epoch

    def parse(self, path):
        print(f"Parsing {path}")
        epoch_wps_regex = re.compile(
            ".*INFO \| train \| epoch (?P<epoch>[0-9]*).* \| wps (?P<wps>[0-9]*).*"
        )
        global_batch_size_regex = re.compile(
            ".*train_batch_size ............. (?P<global_batch_size>[0-9]*).*"
        )

        epoches = []
        wpss = []
        global_batch_size = None

        with open(file=path, mode="r", encoding="UTF-8") as f:
            for line in f:
                match = epoch_wps_regex.match(line)
                if match:
                    epoch = match["epoch"]
                    epoches.append(epoch)
                    wps = float(match["wps"])
                    wpss.append(wps)
                match = global_batch_size_regex.match(line)
                if match:
                    global_batch_size = int(match["global_batch_size"])

        metrics = OrderedDict()

        metrics["global_batch_size"] = global_batch_size
        if len(epoches) == len(wpss):
            if self.select_epoch < len(epoches):
                metrics["performance"] = wpss[self.select_epoch]
            else:
                metrics[f"performance_epoch"] = None
        metrics["metrics"] = "tokens/sec"
        metrics["performance_step"] = self.select_epoch
        print(metrics)

        return metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logpath", type=str, default="moe.log", help="log path"
    )
    parser.add_argument("--select_epoch", type=int, default=0, help="the first step")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    parser = MoeLogParser(select_epoch=args.select_epoch)
    logfile = args.logpath

    metric = parser.parse(logfile)

    labels = list(metric.keys())
    try:
        with open(f"metrics.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=labels)
            writer.writeheader()
            writer.writerow(metric)
    except IOError:
        print("I/O error")

