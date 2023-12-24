
import os
import ants
import numpy as np
from countless3d import countless3d, countless_generalized, dynamic_countless_generalized, countlessND

def flip_image(ants_image, axis=0, single_components=False): 
    data = ants_image.numpy(single_components=single_components)
    flipped_data = np.flip(data, axis=axis)
    if flipped_data.dtype == 'int16':
        flipped_data = flipped_data.astype('float32')
    flipped_data = flipped_data.squeeze()
    return ants_image.new_image_like(flipped_data)

def downsample_images_in_folder(input_folder_path, output_folder_path, factor):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)
    
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder_path):
        # Construct full file path for the input image
        input_file_path = os.path.join(input_folder_path, filename)

        if "LT" in filename.upper(): 
             flip = True
        else: flip = False

        if "SEGMENTATION" in filename.upper(): 
            seg = True
        else: seg = False
        
        # Check if it's a file
        if os.path.isfile(input_file_path):
            # Load the image
            try:
                target_image = ants.image_read(input_file_path)

                if flip:
                    if seg:
                        target_image = flip_image(target_image, axis=0, single_components=True)
                    else:
                        target_image = flip_image(target_image)

                if factor == 1:
                    downsampled_image = target_image
                else: 
                    # Apply the downsampling function
                    origin = target_image.origin
                    spacing = tuple(element * 2 for element in target_image.spacing)
                    direction =  target_image.direction

                    img_np = target_image.numpy()
                    if img_np.shape[2] % 2 == 1:
                        img_np = img_np[:,:,0:-1]

                    downsampled_image = countlessND(img_np, (factor,factor,factor))
                    downsampled_image = ants.from_numpy(downsampled_image, origin = origin, spacing = spacing,
                                                        direction=direction)

                # downsampled_image = ants.resample_image(target_image, [int(size / factor) for size in target_image.shape], 1, 0)
                

                # Construct new filename with downsample size appended
                new_filename = f"{filename.split('.')[0]}_ds{int(512/factor)}.nii.gz"
                output_file_path = os.path.join(output_folder_path, new_filename)

                # Save the downsampled image
                ants.image_write(downsampled_image, output_file_path)
                print(f"Downsampled and saved: {new_filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

# Example usage
OG_size = 512
factor = 1  # This variable isn't used, the countless3d algo is hardcoded as a dowsample size of 2, there is a dynamic method
            # but it's slower

input_folder_path_img = '../../nii_images'  
output_folder_path_img = input_folder_path_img + str(int(OG_size/factor))
downsample_images_in_folder(input_folder_path_img, output_folder_path_img, factor)

input_folder_path_seg = '../../nii_segmentations'  
output_folder_path_seg = input_folder_path_seg + str(int(OG_size/factor))
downsample_images_in_folder(input_folder_path_seg, output_folder_path_seg, factor)