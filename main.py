import numpy as np
import nibabel as nib
import argparse
from datetime import datetime
import logging.config
import json

# TODO: convert to 3D,
# TODO: use poetry to handle the package 

# TODO: add logger
# TODO: fix pixel spacing (https://simpleitk.readthedocs.io/en/v1.2.4/Documentation/docs/source/fundamentalConcepts.html) and data type issue



def get_logger():
    
    with open('./config/log_config.json') as f:
        config = json.load(f)
    
    logging.config.dictConfig(config)

    # Get the logger and use it as before
    logger = logging.getLogger(__name__)
    logger.debug('This is a debug message from dictionary')
    return logger


def is_nifti_file(filename):
    """
    Check if the given filename ends with '.nii' or '.nii.gz'

    :param filename: The filename to check.
    :return: True if the filename ends with '.nii' or '.nii.gz', False otherwise.
    """
    return filename.endswith('.nii') or filename.endswith('.nii.gz')


def get_orientation_data(image: nib.imageclasses, thickness: int, overlap: int, orientation: str) -> dict:
    """_summary_

    Args:
        image (nib.imageclasses): _description_
        orientation (str): _description_

    Returns:
        dict: _description_
    """
    orientation_index_mapping = {
        "sagittal" : 0,
        "coronal" : 1,
        "axial" : 2,
    }
    
    orientation_data = {}
    
    
    orientation_index = orientation_index_mapping.get(orientation)
    
    # get necessary info 
    image_shape = image.header.get_data_shape()
    voxel_size = image.header.get_zooms()[orientation_index]
    
    logging.info(f"Original image shape: {image_shape}")
    
    step_size = thickness - overlap
    pad_size = round(thickness / voxel_size) # 
    
    slice_number = image_shape[orientation_index]
    num_slabs = int(np.ceil((slice_number * voxel_size) / step_size))
    
    image_affine = image.affine
    new_spacing = slice_number * voxel_size / num_slabs
    image_affine[orientation_index, orientation_index] = new_spacing
    
    
    # create a zero image
    if orientation_index == 0:
        projected_image_shape = (num_slabs, image_shape[1], image_shape[2])
        pad_with_dim = ((0, pad_size), (0, 0), (0, 0))
    
    elif orientation_index == 1:
        projected_image_shape = (image_shape[0], num_slabs, image_shape[2])
        pad_with_dim = ((0, 0), (0, pad_size), (0, 0))
    else:
        projected_image_shape = (image_shape[0], image_shape[1], num_slabs)
        pad_with_dim = ((0, 0), (0, 0), (0, pad_size))
        
    
    orientation_data["orientation_index"] = orientation_index
    orientation_data["voxel_size"] = voxel_size
    orientation_data["step_size"] = step_size
    orientation_data["slice_number"] = slice_number
    orientation_data["image_affine"] = image_affine
    orientation_data["projected_image_shape"] = projected_image_shape
    orientation_data["pad_with_dim"] = pad_with_dim
    orientation_data["num_slabs"] = num_slabs
    orientation_data["pad_size"] = pad_size
    
    logging.info(f"orientation_data looks like:\n{orientation_data}")
    return orientation_data


def perform_axial_projection(image_directory: str, thickness: int, overlap: int, projection_type: str, orientation: str):
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
    
    # sanitise the input first, use choice in arg
    if  not is_nifti_file(image_directory):
        raise ValueError("Invalid image directory, please ubderstand this function support nii only")
    
    if not isinstance(thickness, int):
        thickness = int(thickness)
    
    if not isinstance(overlap, int):
        overlap = int(overlap)
      
    logger = get_logger()
    logger.info(f"Starting execute {projection_type} algorithm")

    # get the projection function
    projection_mapping = {
        "MaxIP" : np.max,
        "MinIP" : np.min    
    }
    
    projection_func = projection_mapping.get(projection_type)
    image = nib.load(image_directory)
    logger.info(f"Loaded image from {image_directory}")


    # get necessary info 
    
    orientation_data = get_orientation_data(
        image=image,
        thickness=thickness,
        overlap=overlap,
        orientation=orientation)
    
    orientation_index = orientation_data.get("orientation_index")
    step_size = orientation_data.get("step_size")
    voxel_size = orientation_data.get("voxel_size")
    image_affine = orientation_data.get("image_affine")
    pad_size = orientation_data.get("pad_size")
    num_slabs = orientation_data.get("num_slabs")
    slice_number = orientation_data.get("slice_number")
    
    # create a zero image
    projected_image_shape = orientation_data.get("projected_image_shape")
    projected_image = np.zeros(projected_image_shape)
    
    images_3d_array = image.get_fdata()

    # pad for further calculation
    images_3d_array = np.pad(
        images_3d_array,
        pad_width=orientation_data['pad_with_dim'],
        mode='symmetric')
    
    # Iterate through each slab to get projection image
    for i in range(num_slabs):
        # Calculate start and end index for the current slab (in voxels)
        start_index = round(i * step_size / voxel_size)
        end_index = min(start_index + pad_size, slice_number)

        # TODO: migh over calculate the slab, the process shall stop when reach the last slice of the array
        # Select data for the current slab (considering potential array edge)
        logger.info(f"Calculating projection along {orientation_index} index")
        logger.info(f"The rannge of slab data from {start_index} to {end_index}")
        if orientation_index == 0:
            slab_data = images_3d_array[start_index:end_index, :, :]
        elif orientation_index == 1:
            slab_data = images_3d_array[:, start_index:end_index, :]
        else:
            slab_data = images_3d_array[:, :, start_index:end_index]

        # Take the maximum value within the axial slab
        mip_slice = projection_func(slab_data, axis=orientation_index)
        logger.info(f"Projection along axis {orientation_index} completed")
        
        
        if orientation_index == 0:
            projected_image[i, :, :] = mip_slice
        elif orientation_index == 1:
            projected_image[:, i, :] = mip_slice
        else:
            projected_image[:, :, i] = mip_slice
        logger.info(f"Replace {i}-th slab completed")
    
    
    projected_image = nib.Nifti1Image(projected_image, affine=image_affine)
    
    return projected_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--thickness', type=int, required=True)
    parser.add_argument('--overlap', type=int, required=True)
    parser.add_argument('--projection', type=str, required=True, choices=["MaxIP", "MinIP"])
    parser.add_argument('--orientation', type=str, choices=["axial", "coronal", "sagittal"])
    args = vars(parser.parse_args())
    

    image_path = args["image_path"]
    projection_type = args["projection"]
    thickness = args["thickness"]
    overlap = args["overlap"]
    orientation = args["orientation"]
    
    projected_images = perform_axial_projection(
        image_directory=image_path,
        thickness=thickness, 
        overlap=overlap,
        projection_type=projection_type,
        orientation=orientation
        )
    
    projected_images.to_filename(f"./{projection_type}_thickness_{thickness}_overlap_{overlap}_{orientation}.nii")
    

if __name__ == "__main__":
    main()