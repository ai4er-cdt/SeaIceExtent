import numpy as np
import os

program_path = os.getcwd()
temp_folder = r"{}\temp\temporary_files".format(program_path)
temp_buffer = r"{}\temp\temporary_buffer".format(program_path)
temp_prediction = r"{}\temp\current_prediction".format(program_path)
model_sar = r"{}\models\sar_model_example.pth".format(program_path)
model_modis = r"{}\models\modis_model_example.pth".format(program_path)

def name_file(out_name, file_type, out_path = "temp"):
    """Construct the full path for a new file.
       Parameters: out_path: (string) the path to the folder in which to place the new item, or "temp" or "buffer"
                   to store it temporarily with the program files for the duration of the run-time.
                   out_name: (string) the name of the new file. 
                   file_type: (string) the file extention on the new file.
       Returns: file_name: (string) the full path of the new file.
    """
    if out_path == "temp":
        out_path = temp_folder
    elif out_path == "buffer":
        out_path = temp_buffer
    elif out_path == "prediction":
        out_path = temp_prediction
    file_name = "{}\{}{}".format(out_path, out_name, file_type)
    return file_name


def delete_temp_files():
    """Remove temporary files when no longer needed.
    """
    for folder in [temp_folder, temp_buffer, temp_prediction]:
        os.chdir(folder)
        for temp_file in os.listdir():
            os.remove(temp_file)