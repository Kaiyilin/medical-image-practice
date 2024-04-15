import numpy as np
import nibabel as nib
import argparse
from datetime import datetime
import logging.config
import json

# TODO: use poetry to handle the package 
# TODO: fix data type inconsistent

def get_logger():
    """_summary_

    Returns:
        _type_: _description_
    """
    
    with open('./config/log_config.json') as f:
        config = json.load(f)
    
    logging.config.dictConfig(config)

    # Get the logger and use it as before
    logger = logging.getLogger(__name__)
    logger.info('Logger initiated')
    return logger

logger = get_logger()

def is_nifti_file(filename) -> bool:
    """
    Check if the given filename ends with '.nii' or '.nii.gz'

    :param filename: The filename to check.
    :return: True if the filename ends with '.nii' or '.nii.gz', False otherwise.
    """
    return filename.endswith('.nii') or filename.endswith('.nii.gz')


def get_essential_data(image: nib.imageclasses, thickness: int, overlap: int, orientation: str) -> dict:
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
    
    essential_data = {}
    
    
    orientation_index = orientation_index_mapping.get(orientation)
    
    # get necessary info 
    image_shape = image.header.get_data_shape()
    voxel_size = image.header.get_zooms()[orientation_index]
    
    step_size = thickness - overlap
    delta_thickness = round(thickness / voxel_size) # 
    
    slice_number = image_shape[orientation_index]
    num_slabs = int((slice_number * voxel_size) // step_size) # ensure it's integers
    
    image_affine = image.affine
    new_spacing = slice_number * image_affine[orientation_index, orientation_index] / num_slabs
    image_affine[orientation_index, orientation_index] = new_spacing
    
    # projected_image_shape = list(image_shape)
    # projected_image_shape[orientation_index] = num_slabs
    # projected_image_shape = tuple(projected_image_shape)
    
    # create a zero image. faster
    if orientation_index == 0:
        projected_image_shape = (num_slabs, image_shape[1], image_shape[2])
    elif orientation_index == 1:
        projected_image_shape = (image_shape[0], num_slabs, image_shape[2])
    else:
        projected_image_shape = (image_shape[0], image_shape[1], num_slabs)
        
    essential_data["orientation_index"] = orientation_index
    essential_data["voxel_size"] = voxel_size
    essential_data["step_size"] = step_size
    essential_data["slice_number"] = slice_number
    essential_data["image_affine"] = image_affine
    essential_data["projected_image_shape"] = projected_image_shape
    essential_data["num_slabs"] = num_slabs
    essential_data["delta_thickness"] = delta_thickness

    return essential_data


def perform_projection(image_directory: str, thickness: int, overlap: int, projection_type: str, orientation: str) -> nib.Nifti1Image:
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
        raise ValueError("Invalid image directory, please understand this function support nii only")
    
    if not isinstance(thickness, int):
        thickness = int(thickness)
    
    if not isinstance(overlap, int):
        overlap = int(overlap)
    
    logger.info(f"Starting execute {projection_type} algorithm in {orientation} view")

    # get the projection function
    projection_mapping = {
        "MaxIP" : np.max,
        "MinIP" : np.min    
    }
    
    projection_func = projection_mapping.get(projection_type)
    image = nib.load(image_directory)
    logger.info(f"Loaded image from {image_directory}")


    # get necessary info 
    
    logger.info(f"Getting essential meta data for calculation...")
    essential_data = get_essential_data(
        image=image,
        thickness=thickness,
        overlap=overlap,
        orientation=orientation)
    
    logger.info(f"Got essential meta data: \n {essential_data}")
    
    orientation_index = essential_data.get("orientation_index")
    
    # create a zero image
    projected_image_shape = essential_data.get("projected_image_shape")
    projected_image = np.zeros(projected_image_shape)
    logger.info(f"Created Zero image with shape: {projected_image_shape}")
    
    images_3d_array = image.get_fdata()
    
    logger.info(f"Starting calculating {projection_type}...")
    # Iterate through each slab to get projection image
    for i in range(essential_data.get("num_slabs")):
        # Calculate start and end index for the current slab (in voxels)
        start_index = round(i * essential_data.get("step_size") / essential_data.get("voxel_size"))
        end_index = min(start_index + essential_data.get("delta_thickness"),  essential_data.get("slice_number"))

        # Select data for the current slab (considering potential array edge)
        logger.info(f"Get the slab data from the slice {start_index} to {end_index} of the original data")
        if orientation_index == 0:
            slab_data = images_3d_array[start_index:end_index, :, :]
        elif orientation_index == 1:
            slab_data = images_3d_array[:, start_index:end_index, :]
        else:
            slab_data = images_3d_array[:, :, start_index:end_index]

        # Take the maximum value within the axial slab
        mip_slice = projection_func(slab_data, axis=orientation_index)
        logger.info(f"Projection completed")
        
        logger.info(f"Starting replace {i}-th zero image...")
        if orientation_index == 0:
            projected_image[i, :, :] = mip_slice
        elif orientation_index == 1:
            projected_image[:, i, :] = mip_slice
        else:
            projected_image[:, :, i] = mip_slice
        logger.info(f"Replaced {i}-th zero image")
    
    logger.info(f"Finished {projection_type} image calculation on {orientation} view")
    projected_image = nib.Nifti1Image(projected_image, affine=essential_data.get("image_affine"))
    
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
    
    projected_images = perform_projection(
        image_directory=image_path,
        thickness=thickness, 
        overlap=overlap,
        projection_type=projection_type,
        orientation=orientation
        )
    
    projected_images.to_filename(f"./{projection_type}_thickness_{thickness}_overlap_{overlap}_{orientation}.nii")
    logger.info(f"projection image saved to ./{projection_type}_thickness_{thickness}_overlap_{overlap}_{orientation}.nii\n")

if __name__ == "__main__":
    main()