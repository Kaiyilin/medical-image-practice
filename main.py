import numpy as np
import nibabel as nib
import argparse
from datetime import datetime


def is_nifti_file(filename):
    """
    Check if the given filename ends with '.nii' or '.nii.gz'

    :param filename: The filename to check.
    :return: True if the filename ends with '.nii' or '.nii.gz', False otherwise.
    """
    return filename.endswith('.nii') or filename.endswith('.nii.gz')


def perform_axial_projection(image_directory: str, thickness: int, overlap: int, projection_type: str):
    """_summary_

    Args:
        image_path (str): path to nii file, currently support .nii extension only
        thickness (int): an integer that defines the slab thickness
        overlap (int): an interger for step size
        projection_type (str): define which projection you'd like to use MaxIP or MinIP

    Raises:
        ValueError: _description_
    
    Return:
        A nibabel nifti object
    """
    
    # sanitise the input first 
    if projection_type not in ("MaxIP", "MinIP"):
        raise ValueError("Invalid prjection type, please select MaxIP or MinIP")
    
    if  not is_nifti_file(image_directory):
        raise ValueError("Invalid image directory, please ubderstand this function support nii only")
    
    if not isinstance(thickness, int):
        thickness = int(thickness)
    
    if not isinstance(overlap, int):
        overlap = int(overlap)
        
    
    
    # get the projection function
    projection_mapping = {
        "MaxIP" : np.max,
        "MinIP" : np.min    
    }
    
    projection_func = projection_mapping.get(projection_type)
    image = nib.load(image_directory)

    # get necessary info 
    step_size = thickness - overlap
    voxel_size = image.header.get_zooms()[-1]
    pad_size = round(thickness / voxel_size)
    
    images_3d_array = image.get_fdata()
    axial_slice_number = images_3d_array.shape[-1]
    num_slabs = int(np.ceil((axial_slice_number * voxel_size) / step_size))
    
    image_affine = image.affine
    
    # create a zero image
    projected_image_shape = (images_3d_array.shape[0], images_3d_array.shape[1], num_slabs)
    projected_image = np.zeros(projected_image_shape)

    # pad for further calculation
    images_3d_array = np.pad(
        images_3d_array,
        pad_width=((0, 0), (0, 0), (0, pad_size)),
        mode='symmetric')
    
    # Iterate through each slab to get projection image
    for i in range(num_slabs):
        # Calculate start and end index for the current slab (in voxels)
        start_index = round(i * step_size / voxel_size)
        end_index = min(start_index + pad_size, axial_slice_number)

        # Select data for the current slab (considering potential array edge)
        slab_data = images_3d_array[:, :, max(0, start_index):end_index]

        # Take the maximum value within the axial slab
        mip_slice = projection_func(slab_data, axis=-1)
        
        projected_image[:,:, i] = mip_slice
    
    
    # TODO: removed when the main function is completed
    projected_image = nib.Nifti1Image(projected_image, affine=image_affine)
    
    return projected_image
    
    
     

dicom_files_dir = '/Users/kaiyilin/Documents/code_project/medical-image-practice/Input/3D_AXIAL_SWI_0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--thickness', type=int, required=True)
    parser.add_argument('--overlap', type=int, required=True)
    parser.add_argument('--projection', type=str, required=True)
    
    args = vars(parser.parse_args())
    
    image_path = args["image_path"]
    projection_type = args["projection"]
    thickness = args["thickness"]
    overlap = args["overlap"]
    
    projected_images = perform_axial_projection(
        image_directory=image_path,
        thickness=thickness, 
        overlap=overlap,
        projection_type=projection_type
        )
    
    projected_images.to_filename(f"./{projection_type}_thickness_{thickness}_overlap_{overlap}_{datetime.now()}.nii")
    

if __name__ == "__main__":
    main()