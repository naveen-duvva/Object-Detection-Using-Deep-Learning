import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


# Detect the model
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
filename = 'labels.names'
with open(filename, 'rt') as spt:
    classLabels = spt.read().rstrip('\n').split('\n')
    
    
model.setInputSize(320, 320) #greater this value better the reults tune it for best output 320, 320
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

file_path=''

# Create a function to open an image using a file dialog and resize it to 512x512 pixels
def open_image():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ppm *.pgm")])
    if file_path:
        original_image = Image.open(file_path)
        resized_image = original_image.resize((512, 512))
        photo = ImageTk.PhotoImage(resized_image)
        frame1.config(image=photo)
        frame1.image = photo

# Create a function to process the selected image and display it in the second frame
def process_image():
    if frame1.image:
        numpy_image = detect_objects(file_path)
        #convert a numpy array to a format that a tkinter can use it as a image
        sample_bgr = cv2.cvtColor(numpy_image,cv2.COLOR_RGB2BGR) #OpenCV uses BGR, so we convert from RGB
        sample_img=Image.fromarray(sample_bgr)
        resized_img=sample_img.resize((512,512))
        processed_image = ImageTk.PhotoImage(resized_img)
        frame2.config(image=processed_image)
        frame2.image = processed_image

def detect_objects(path):
    img = cv2.imread(path)
    
    classIndex, confidence, bbox = model.detect(img, confThreshold=0.5) #tune confThreshold for best results


    font = cv2.FONT_HERSHEY_PLAIN

    for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
        cv2.rectangle(img, boxes, (255, 0, 0), 2)
        cv2.putText(img, classLabels[classInd-1], (boxes[0] + 10, boxes[1] + 40), font, fontScale = 3, color=(0, 255, 0), thickness=3)
    return img

# Create the main application window
root = tk.Tk()
root.title("Object Detection From Images")

# Maximize the window by default
root.state('zoomed')

root.configure(bg='#ffc0cb')

# Create two frames for displaying images
frame1 = tk.Label(root)
frame2 = tk.Label(root)

frame1.grid(row=0, column=0, padx=10, pady=10)
frame2.grid(row=0, column=1, padx=10, pady=10)

# Create "Open Image" button
open_button = tk.Button(root, text="Select Image", command=open_image)
open_button.grid(row=1, column=0, padx=10, pady=10)

# Create "Process Image" button
process_button = tk.Button(root, text="Detect", command=process_image)
process_button.grid(row=1, column=1, padx=10, pady=10)

# Create a blank 512x512 image
blank_image = Image.new("RGB", (512, 512), "#ffc0cb")
blank_photo = ImageTk.PhotoImage(blank_image)

# Display the blank image in the frames by default
frame1.config(image=blank_photo)
frame1.image = blank_photo
frame2.config(image=blank_photo)
frame2.image = blank_photo

root.mainloop()
