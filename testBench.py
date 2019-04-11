import time
import numpy as np
from PIL import Image
from rockethub import Rocket
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import resource

import os, platform, subprocess

def average_time_inference(model: nn.Module, input: Image, num_iterations: int = 10, device: str = 'cpu') -> int:
    """Compute the average time for the inference of the model.

    """
    with torch.no_grad():
        # Pre-process the image
        img_tensor = model.preprocess(img).to(device)
        # Warmup the model
        out = model(img_tensor)

        arr_time = np.zeros(num_iterations)
        
        # Testing Loop
        pbar = tqdm(total=num_iterations, ascii=True, desc='Rocket Testing')
        for i in range(num_iterations):
            start_inference = time.time()
            
            out = model(img_tensor)

            if device == 'cuda' : torch.cuda.synchronize() #To synchronise GPU with CPU

            arr_time[i] = time.time() - start_inference
            
            pbar.update(1)
        pbar.close()
    
    return np.mean(arr_time)

def sizeof_fmt(num, suffix='B'):
    """Convert bit size to a human readable format
    
    Source: https://stackoverflow.com/a/1094933/1568937
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def get_cpu_name() -> str:
    """ Get the CPU name of the machine where the model is run.

    """
    if platform.system() == "Linux":
        command = ["cat", "/proc/cpuinfo"]
        all_info = str(subprocess.Popen(command, stdout=subprocess.PIPE ).communicate()[0]).strip()
        start_str = all_info.find('model name') + len('model_name') + 4
        end_str = start_str + all_info[start_str:].find('\\n')
        model_name = all_info[start_str:end_str]
        return model_name
    else:
        print('The cpu name can only be retrieved on Linux.')
        return 'Unknown'

def display_test_results(test_results: dict, width: int = 55):
    """ Display the results in the terminal

    """
    # Compute potential longuest string
    max_key_width = max([len(str(v[0])) for v in test_results])
    # max_item_width = max([len(str(v[1])) for v in test_results])
    # print(max_key_width, max_item_width)

    # Print the results
    print((' TEST RESULTS ').center(width, '-'))
    for v in test_results:
        line = '| ' + v[0].ljust(max_key_width) + ' : ' + v[1].ljust(width)
        print(line[:width-2], '|')
    print(('').center(width, '-'))

def convert_results2dict(test_results: dict):
    """ Convert the list of results to a python Dictionary

    """
    return {v[0]:v[1] for v in test_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the inference speed of a Rocket.")
    parser.add_argument('--rocket', type=str, help="Name of the Rocket to test.")
    parser.add_argument('--image', type=str, help="Path to the image to use for the test.")
    parser.add_argument('--iterations', type=int, default=10, help="Number of iterations in the testing loop.")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help="Device on wich to use the model.")
    parser.add_argument('--unit', type=str, choices=['sec', 'FPS'], default='FPS', help="Unit in which to display the speed of the Rocket.")
    opt = parser.parse_args()

    # Initiate the dict to gather all the test results
    test_results = []

    # Load the image
    img = Image.open(opt.image)

    # Optimize for the GPU
    if opt.device == 'cuda':
        torch.cuda.benchmark=True

    # Load the model
    model = Rocket.land(opt.rocket).to(opt.device).eval()

    # Get the number of parameters
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    test_results.append(['Rocket', opt.rocket])
    test_results.append(['#Param', "{:,}".format(num_parameters)])
    test_results.append(['Device', opt.device])
    

    # Starting the Testing
    out = average_time_inference(model, img, num_iterations=opt.iterations, device=opt.device)
    
    if platform.system() == "Linux":
        test_results.append(['CPU', get_cpu_name()])
        # Doesn't seem to provide the metrics we want
        # test_results.append(['CPU Usage', sizeof_fmt(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)]) 

    if opt.device == 'cuda':
        test_results.append(['GPU', torch.cuda.get_device_name(0)])
        test_results.append(['GPU Usage', sizeof_fmt(torch.cuda.max_memory_allocated(0))])

    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    test_results.append(['Image', opt.image])
    # Get speed in the right format
    out = round(out, 3) if opt.unit == 'sec' else round(1/out, 3)

    test_results.append(['Speed', str(out) + ' ' + opt.unit])

    display_test_results(test_results)

    # Convert the test results to dict for .jsom
    # print(convert_results2dict(test_results))
