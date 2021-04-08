import os
import logging

import parsl
from parsl.config import Config
from parsl.executors.ipp import IPyParallelExecutor
from libsubmit.providers import LocalProvider
from libsubmit.providers import SlurmProvider
from libsubmit.launchers import SrunLauncher
from libsubmit.channels import LocalChannel

def setup_parsl(machine, n_slots, n_thread=1, walltime="00:30:00", memory=16000, parsl_dir="./parsl", partition="all"):
    parsl.clear()
    if machine == "local":
        script_dir = os.path.join(parsl_dir, "parsl_scripts")
        run_dir = os.path.join(parsl_dir, "runinfo")

        local_ipp_config = Config(
                                executors=[
                                    IPyParallelExecutor(
                                        label="local_ipp",
                                        provider=LocalProvider(
                                            channel=LocalChannel(userhome=parsl_dir, script_dir=parsl_dir),
                                            init_blocks=1,
                                            max_blocks=n_slots,
                                            script_dir=parsl_dir,
                                        ),
                                        engine_dir=parsl_dir,
                                        working_dir=parsl_dir,
                                    )
                                ],
                                app_cache=False,
                                run_dir=run_dir,
                                lazy_errors=False,
                            )
        parsl.load(local_ipp_config)
        parsl.set_stream_logger(level=logging.INFO)
    elif machine == "cuillin":
        script_dir = os.path.join(parsl_dir, "parsl_scripts")
        run_dir = os.path.join(parsl_dir, "runinfo")
 
        slurm_overrides = "#SBATCH --mem-per-cpu {memory}\n#SBATCH --cpus-per-task {n_thread}\n" \
                          "#SBATCH --constraint=datadisk".format(memory=memory, n_thread=n_thread)
        cuillin_ipp_config = Config(
                                    executors=[
                                        IPyParallelExecutor(
                                            label="cuillin_ipp",
                                            provider=SlurmProvider(
                                                partition=partition,
                                                channel=LocalChannel(userhome=parsl_dir, script_dir=parsl_dir),
                                                launcher=SrunLauncher(),
                                                walltime=walltime,
                                                init_blocks=1,
                                                max_blocks=n_slots,
                                                script_dir=parsl_dir,
                                                overrides=slurm_overrides,
                                            ),
                                            working_dir=parsl_dir,
                                        )
                                    ],
                                    app_cache=False,
                                    run_dir=run_dir,
                               )
        parsl.load(cuillin_ipp_config)
        parsl.set_stream_logger(level=logging.INFO)
    else:
        raise ValueError(f"Machine '{machine}' not supported.")

    os.makedirs(parsl_dir, exist_ok=True)
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    return parsl_dir
