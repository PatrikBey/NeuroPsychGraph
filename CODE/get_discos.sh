




files=$( seq 1 1000)




masks="Glioma Hemorrhagic Ischaemic"

for m in $masks; do
    mkdir -p "${m}_clean"
    for path in "${m}"/*.nii "${m}"/*.nii.gz; do
        [ -f "$path" ] || continue
        f=$(basename "$path")
        fslroi "$path" "${m}_clean/${f}" 0 1
    done
done



for m in $masks; do
    mkdir -p "${m}_clean"
    for path in "${m}"/*.nii "${m}"/*.nii.gz; do
        [ -f "$path" ] || continue
        f=$(basename "$path")
        fslroi "$path" "${m}_clean/${f}" 0 1
    done
done




Path="/data/patrik/NPG/DATA"
masks="Glioma Hemorrhagic Ischaemic"

max_jobs=20
cpus=3

for m in $masks; do
    files=$(ls ${Path}/LESIONS/${m})
    mkdir -p ${Path}/DISCONNECTOMES/${m}

    for f in ${files}; do
        (
            docker run --rm --cpus="${cpus}" --user "$(id -u):$(id -g)" \
                -v ${Path}/LESIONS/${m}:/data \
                -v ${Path}/DISCONNECTOMES/${m}:/output \
                -e Seed=${f} \
                -e Atlas="Schaefer2018-100" \
                -e OutDir=/output \
                -e tck_keep=False \
                arise:0.6
        ) &

        while [ "$(jobs -rp | wc -l)" -ge "${max_jobs}" ]; do
            wait -n
        done
    done
done

wait
