for noise_type in "normal" "lognormal"; do
    for fmodel_type in "MLP"; do
        for kernel_fn in "diff" "none"; do
            for X_type in "normal" "binary"; do 
                name=${noise_type}_${fmodel_type}_${kernel_fn}_${X_type}
                qsub -N $name simu_basic.sh $noise_type $fmodel_type $kernel_fn $X_type
            done
        done
    done
done