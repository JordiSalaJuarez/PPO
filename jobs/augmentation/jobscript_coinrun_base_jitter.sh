 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J ppoRun
 #BSUB -n 1
 #BSUB -W 18:00
 #BSUB -R "rusage[mem=32GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 echo "Running script..."
 poetry run python -m ppo --env_name coinrun --do_jitter --use_backgrounds
