import subprocess
import torch
import os

class _ColorHEX:
    blue = "#1f77b4"
    orange = "#ff7f0e"
    green = "#2ca02c"
    red = "#d62728"
    purple = "#9467bd" 
    brown = "#8c564b"
    pink = "#e377c2"
    grey =  "#7f7f7f"
colors_hex = _ColorHEX()


def get_gpu_memory_info():
    """ Return the systems total amount of VRAM along with current used/free VRAM"""
    # TODO: check if ´nvidia-smi´ is installed 
    # TODO: Enable multi-gpu setup i.e. cuda:0, cuda:1 ...

    
    def get_info(command):
        assert command in ["free", "total"]
        command = f"nvidia-smi --query-gpu=memory.{command} --format=csv"
        info = output_to_list(subprocess.check_output(command.split()))[1:]
        values = [int(x.split()[0]) for i, x in enumerate(info)]
        return values[0]
        
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    free_vram = get_info("free")
    total_vram = get_info("total")
    return {"GPU":torch.cuda.get_device_properties(0).name,
            "free": free_vram, 
            "used": total_vram-free_vram, 
            "total":total_vram
            }
    
def get_gpu_info():
    return {"name" : torch.cuda.get_device_properties(0).name,
            "major" : torch.cuda.get_device_properties(0).major,
            "minor" : torch.cuda.get_device_properties(0).minor,
            "total_memory" : torch.cuda.get_device_properties(0).total_memory/10**6,
            "multi_processor_count" : torch.cuda.get_device_properties(0).multi_processor_count
            }


def write_to_file(file_path:str, write_string:str, only_txt:bool = True):
    """ Appends a string to the end of a file"""
    if only_txt:
        assert extract_file_extension(file_path) == ".txt", "´only_txt´ = true, but file type is not .txt"
    
    file = open(file_path, mode="a")
    print(write_string, file=file, end="")
    file.close()