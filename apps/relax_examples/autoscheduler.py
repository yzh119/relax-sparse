import os
import argparse
from tvm.relay import testing
import tvm
from tvm import relay, auto_scheduler


network_to_n_trials = {
    # CPU
    ("resnet_50", 1, "float32", "llvm"): 22000,
}


def auto_scheduler_tune(network, batch_size, dtype, target, log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.exists(log_file):
        os.remove(log_file)

    layout = "NHWC"
    mod, params = testing.resnet.get_workload(num_layers=50, batch_size=1, dtype="float32")

    n_trials = network_to_n_trials[(network, batch_size, dtype, str(target.kind))]

    if "cpu" in target.keys:
        print("lol")
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
    else:
        min_repeat_ms = 450 if network in ["bert"] else 300
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            repeat=1, min_repeat_ms=min_repeat_ms, timeout=10
        )
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(tuning_opt)


if __name__ == "__main__":
    target = tvm.target.Target("llvm", host="llvm")
    auto_scheduler_tune("resnet_50", 1, "float32", target, "./tmp_logs/autoscheduler_resnet50.json")