#!/bin/bash
max_folds=30
start_fold=2
maxNumSubmitted=700
queue="compute-16-64,compute-24-128"
username="eej17ucu"
mail="NONE"
mailto="eej17ucu@uea.ac.uk"
max_memory=8000
max_time="168:00:00"
start_point=1
data_dir="/gpfs/home/ajb/Data/"
results_dir="/gpfs/home/eej17ucu/clustering-results/averaging-experiment/"
out_dir="/gpfs/home/eej17ucu/code/output/"
script_file_path="/gpfs/home/eej17ucu/code/sktime/sktime/_contrib/clustering_experiments/averaging/averaging_experiments.py"
env_name="sktime"
datasets="/gpfs/home/ajb/DataSetLists/TSC_112_2019.txt"
generate_train_files="true"
predefined_folds="true"
clusterer="kmeans"
count=0
while read dataset; do
  for distance in euclidean
  do
    numPending=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "PD ${queue}" | wc -l)
    numRunning=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "R ${queue}" | wc -l)
    while [ "$((numPending+numRunning))" -ge "${maxNumSubmitted}" ]
    do
        echo Waiting 60s, $((numPending+numRunning)) currently submitted on ${queue}, user-defined max is ${maxNumSubmitted}
      sleep 60
      numPending=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "PD ${queue}" | wc -l)
      numRunning=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "R ${queue}" | wc -l)
    done

    ((count++))

    if ((count>=start_point)); then

    mkdir -p ${out_dir}${classifier}/${dataset}/
    echo "#!/bin/bash

    #SBATCH --mail-type=${mail}
    #SBATCH --mail-user=${mailto}
    #SBATCH -p ${queue}
    #SBATCH -t ${max_time}
    #SBATCH --job-name=${classifier}${dataset}
    #SBATCH --array=${start_fold}-${max_folds}
    #SBATCH --mem=${max_memory}M
    #SBATCH -o ${out_dir}${classifier}/${dataset}/%A-%a.out
    #SBATCH -e ${out_dir}${classifier}/${dataset}/%A-%a.err

    . /etc/profile

    module add python/anaconda/2019.10/3.7
    source /gpfs/software/ada/python/anaconda/2019.10/3.7/etc/profile.d/conda.sh
    conda activate $env_name
    export PYTHONPATH=$(pwd)

    python ${script_file_path} ${data_dir} ${results_dir} ${distance} ${dataset} ${max_folds} ${generate_train_files} ${predefined_folds} ${clusterer}"  > generatedFile.sub

    echo ${count} ${classifier}/${dataset}

    sbatch < generatedFile.sub --qos=ht
    fi

  done
done < ${datasets}

echo Finished submitting jobs
