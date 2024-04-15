dicom_files_dir='/Users/kaiyilin/Documents/code_project/medical-image-practice/Input/3D_AXIAL_SWI_0.nii.gz'
thickness=5
overlap=3
projection_type='MaxIP'
orientations=(
    'axial'
    'coronal'
    'sagittal'
    )


for orientation in "${orientations[@]}"; do
    python main.py --image_path $dicom_files_dir --thickness $thickness --overlap $overlap --projection $projection_type --orientation $orientation
done
