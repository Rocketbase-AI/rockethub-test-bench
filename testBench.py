import time
import numpy as np
from PIL import Image
from rockethub import Rocket
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

def average_time_inference(model: nn.Module, input: Image, num_iterations: int = 10):
    with torch.no_grad():
        # Pre-process the image
        img_tensor = model.preprocess(img)
        # Warmup the model
        out = model(img_tensor)

        arr_time = np.zeros(num_iterations)
        
        # Testing Loop
        pbar = tqdm(total=num_iterations, ascii=True, desc='Rocket Testing')
        for i in range(num_iterations):
            start_inference = time.time()
            
            out = model(img_tensor)
            
            arr_time[i] = time.time() - start_inference
            pbar.update(1)
        pbar.close()
    
    return np.mean(arr_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the inference speed of a Rocket.")
    parser.add_argument('--rocket', type=str, help="Name of the Rocket to test.")
    parser.add_argument('--image', type=str, help="Path to the image to use for the test.")
    parser.add_argument('--iterations', type=int, default=10, help="Number of iterations in the testing loop.")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help="Device on with to use the model.")
    parser.add_argument('--unit', type=str, choices=['sec', 'FPS'], default='FPS', help="Unit in which to measure the speed of the Rocket.")
    opt = parser.parse_args()

    img = Image.open(opt.image)

    model = Rocket.land(opt.rocket).to(opt.device).eval()

    # Starting the Testing
    out = average_time_inference(model, img, num_iterations=opt.iterations)
    
    # Get speed in the right format
    out = round(out, 3) if opt.unit == 'sec' else round(1/out, 3)

    # Get the number of parameters
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('------ TEST RESULTS ------')
    print('| Rocket: ' + str(opt.rocket).ljust(14) + ' |') 
    print('| Device: ' + str(opt.device).ljust(14) + ' |')
    print('| #Param: ' + ("{:,}".format(num_parameters)).ljust(14) + ' |')
    print('| Image:  ' + str(opt.image).ljust(14) + ' |')
    print('| Speed:  ' + (str(out) + ' ' + opt.unit).ljust(14) + ' |')
    print('--------------------------')
