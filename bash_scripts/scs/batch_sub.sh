for noise_type in "normal" "lognormal"; do
    for fmodel_type in "MLP"; do
        for kernel_fn in "none"; do
            for X_type in "normal" "binary"; do 
                name=simu_${noise_type}_${fmodel_type}_${kernel_fn}_${X_type}
                sbatch -J $name simu_basic.sh $noise_type $fmodel_type $kernel_fn $X_type
            done
        done
    done
done