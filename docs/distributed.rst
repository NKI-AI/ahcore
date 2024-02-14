Distributed Training / Inference
================================

Ahcore is fully compatible with distributed training. Here, we show basic commands to get started with multi-GPU trainings.
Since Ahcore is based on Lightning, we can change the configuration for the Trainer; we can, for example, use a setup such as (which is similar to the ``default_ddp.yaml`` provided in the standard ahcore configs):

.. code-block:: YAML

    _target_: pytorch_lightning.Trainer

    accelerator: gpu
    devices: 2
    num_nodes: 1
    max_epochs: 1000
    strategy: ddp
    precision: 32

which will execute on 2 GPUs (devices=2) on 1 node using Distributed Data Parallel (DDP). Launching from the command line can be done using, e.g., torch distributed launch:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=2 --use_env /.../ahcore/tools/train.py data_description=something lit_module=your_module trainer=default_ddp

Note that a simple command without a distributed launch might only detect 1 GPU!

More commonly, Ahcore distributed can be called using SLURM by sbatch files, for instance:

.. code-block:: bash

    #!/bin/bash

    #SBATCH --job-name=train_ahcore_distributed
    #SBATCH --output=%x_%j.out
    #SBATCH --error=%x_%j.err
    #SBATCH --partition=your_partition
    #SBATCH --qos=your_qos
    #SBATCH --tasks-per-node=2 # Set equal to number of gpus, See comments below
    #SBATCH --gres=gpu:2
    #SBATCH --cpus-per-task=16 # will be multiplied with thie tasks-per-node
    #SBATCH --mem=100G # Adjust memory to your requirement
    #SBATCH --time=12:00:00 # Adjust maximum time (HH:MM:SS)

    # Activate your virtual environment if needed
    source activate /path/to/your/env/

    # Run the training script using srun -- see comments below
    srun python /.../ahcore/tools/train.py \
        data_description=something \
        lit_module=your_module \
        trainer=default_ddp

A few subtleties here: the ``--tasks-per-node`` is introduced for proper communication between Lightning and SLURM, we need to set it equal to the number of gpus. See `here <https://github.com/Lightning-AI/pytorch-lightning/blob/1d04c10e2d26c6097794379f44426cfd78bbd1f1/src/lightning/fabric/plugins/environments/slurm.py#L165/>`_.
Furthermore, the python command is preceded by 'srun', which ensures that environments are properly setup; if we don't add this, the code may hang on initializing the different processes (deadlocked). 
