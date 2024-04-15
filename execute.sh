dicom_files_dir='/Users/kaiyilin/Documents/code_project/medical-image-practice/Input/3D_AXIAL_SWI_0.nii.gz'
thickness=10
overlap=5
projection_type='MaxIP'
orientation='sagittal'


# pyvenvact
python main.py --image_path $dicom_files_dir --thickness $thickness --overlap $overlap --projection $projection_type --orientation $orientation