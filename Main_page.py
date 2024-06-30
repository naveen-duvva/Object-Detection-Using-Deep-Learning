import tkinter as tk 
from PIL import Image,ImageTk
import subprocess

def Live():
    subprocess.call(['python', r'Livecam_Detect.py'])

def Video():
    subprocess.call(['python', r'Video_Detect.py'])
    
def Images():
    subprocess.call(['python',r'Image_Detect.py'])

Main_frame = tk.Tk()
Main_frame.geometry("1235x725")
Main_frame.resizable(False,False)
Main_frame.title("OBJECT DETECTION")
Main_frame.configure(background = 'black') 

Main_frame.grid_rowconfigure(0, weight = 1) 
Main_frame.grid_columnconfigure(0, weight = 1) 

Bg_image = Image.open('Background.png')
Bg_image_p = ImageTk.PhotoImage(Bg_image)
Bg_label = tk.Label(Main_frame,image = Bg_image_p) 
Bg_label.place(x=0,y=0)


sub1img = Image.open('WebCamIcon.png')
sub1img_p = ImageTk.PhotoImage(sub1img)

sub1button = tk.Button(Main_frame,command = Live,image = sub1img_p , bd=0 ,bg='#F6F4F4',activebackground='white')
sub1button.place(x = 700, y = 150)
sub1button_label = tk.Label(Main_frame,text = "Web Cam",fg="black",bg="#0d488f",font=('Gabriola', 20))
sub1button_label.place(x=709, y=252)


sub2img = Image.open('VideoIcon.png')
sub2img_p = ImageTk.PhotoImage(sub2img)

sub2button = tk.Button(Main_frame,command = Video, image = sub2img_p , bd=0 ,bg='#F6F4F4',activebackground='white')
sub2button.place(x = 700, y = 350)
sub2button_label = tk.Label(Main_frame,text = "Video",fg="black",bg="#0d488f",font=('Gabriola', 20))
sub2button_label.place(x=720, y=432)

sub3img = Image.open('ImageIcon.jpg')
sub3img_p = ImageTk.PhotoImage(sub3img)

sub3button = tk.Button(Main_frame,command = Images, image = sub3img_p , bd=0 ,bg='#F6F4F4',activebackground='white')
sub3button.place(x = 700, y = 550)
sub3button_label = tk.Label(Main_frame,text = "Image",fg="black",bg="#0d488f",font=('Gabriola', 20))
sub3button_label.place(x=720, y=632)

Main_frame.mainloop()