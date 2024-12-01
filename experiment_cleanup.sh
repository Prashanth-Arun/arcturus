#!/bin/bash

# NOTE: Parser sourced from https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
POSITIONAL_ARGS=()
SLURM_JOB=./job_script.sh

while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--task)
      TASK_NAME="$2"
      shift # past argument
      shift # past value
      ;;
    -n|--name)
      EXPERIMENT_NAME="$2"
      shift # past argument
      shift # past value
      ;;
     -m|--model)
      MODEL_NAME="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

OUTPUT_PATH=./artifacts/outputs/slurm.out
ERROR_PATH=./artifacts/outputs/slurm.err

OUTPUT_DEST=./artifacts/$TASK_NAME/$MODEL_NAME/$EXPERIMENT_NAME/slurm.out
ERROR_DEST=./artifacts/$TASK_NAME/$MODEL_NAME/$EXPERIMENT_NAME/slurm.err
JOB_DEST=./artifacts/$TASK_NAME/$MODEL_NAME/$EXPERIMENT_NAME/job.sh

mv $OUTPUT_PATH $OUTPUT_DEST
mv $ERROR_PATH $ERROR_DEST
cp $SLURM_JOB $JOB_DEST