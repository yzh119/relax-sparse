import os
from tvm.relay import testing
import tvm
from tvm import relay, auto_scheduler
import numpy as np
from tvm.runtime import vm as vm_rt
import time


def auto_scheduler_tune(batch_size, target, log_file, layout="NHWC", dtype="float32"):
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)

    mod, params = testing.resnet.get_workload(
        num_layers=50,
        batch_size=batch_size,
        dtype=dtype,
        layout=layout,
        image_shape=image_shape,
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.exists(log_file):
        print("tuned file already exists, no need to tune")
    else:
        n_trials = 22000
        print("tuning")

        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
        for idx, task in enumerate(tasks):
            print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
            print(task.compute_dag)

        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tuner.tune(tuning_opt)

    return mod, params


if __name__ == "__main__":
    target = tvm.target.Target("llvm -mcpu=core-avx2 --num-cores=16")
    network = "resnet-50"
    batch_size = 1
    layout = "NCHW"
    # layout = "NHWC"
    dtype = "float32"
    log_file = "autoscheduler_logs/" + "%s-%s-B%d-%s.json" % (
        network,
        layout,
        batch_size,
        target.kind.name,
    )

    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)
    input_shape = (batch_size,) + image_shape

    mod, params = auto_scheduler_tune(batch_size, target, log_file, layout, dtype)

    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            exe = relay.vm.compile(mod, target, params=params)

    input_shape = (1, 3, 224, 224)
    data = tvm.nd.array(np.random.rand(*input_shape).astype(np.float32))

    print("Running 100 inferences...")
    relay_vm = vm_rt.VirtualMachine(exe, tvm.cpu())
    result = relay_vm.run(data)
    tic = time.time()
    for i in range(100):
        relay_vm.run(data)
    toc = time.time()
    e0 = toc - tic
    print(f"relay: {e0}")
