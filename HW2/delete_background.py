import cv2
import numpy as np
from rembg import remove
from PIL import Image

def delete_background():
    input_path = "/Users/petrushovvv/lessons_of_python/HW2/BCCD Dataset with mask/train/image.png"
    output_path = "image_no_background.png"

    with open(input_path, "rb") as i:
        with open(output_path, "wb") as o:
            input_data = i.read()
            output_data = remove(input_data)
            o.write(output_data)

    # Открытие результата
    img = Image.open(output_path)
    img.show()

if __name__=="__main__":
    delete_background()