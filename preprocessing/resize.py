import torchio as tio
import SimpleITK as sitk
import torch

def image_resample(input_path, output_path):
    print('Running Image Resample')
    img_tensor = img_obj_to_array(input_path)
    image = tio.ScalarImage(tensor = img_tensor)
    resize = tio.Resize((128, 128, 128))
    resampled_image = resize(image)
    sitk.WriteImage(resampled_image, output_path)

def img_obj_to_array(input_img):
    print('Running Image To_Array')
    img_numpy = sitk.GetArrayFromImage(input_img)
    torch_tensor = torch.from_numpy(img_numpy).float()
    torch_tensor = torch_tensor.unsqueeze(0)
    return torch_tensor