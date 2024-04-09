import tkinter as tk
import tkinter.messagebox
import Transform_prediction

from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from PIL import ImageTk,Image
from tkinterdnd2 import DND_FILES
from tkinterdnd2 import *

class App(TkinterDnD.Tk):
  def __init__(self):
    super().__init__()

    # configure the root window
    self.title('GUI TER')
    self.geometry('600x450')
    self.filepathimage = ""
    self.radio_var = tkinter.IntVar(value=0)

    # labels
    self.label_radio = ttk.Label(self,text="Modèles:").place(x=400,y=10)
    label1 = tk.Label(self,text="")
    label1.place(x=225,y=125)
    label2 = tk.Label(self,text="")
    label2.place(x=175,y=150)
    
# create radiobutton frame
    # Dictionary to create multiple buttons
    values = {"KNN" : "0",
          "Support Vector Machine" : "1",
          "Linear Support Vector Machine" : "2",
          "Logistic Regression" : "3",
          "Gradient Boosting" : "4",
          "Random Forest" : "5",
          "Multi-Layers Perceptron": "6",
          "Convolution Neural Network": "7"}

    for (text, value) in values.items():
        value_y = 40 + int(value) * 40
        #print(value_y)
        tk.Radiobutton(self.label_radio,text=text, variable=self.radio_var,
                   value=value).place(x=400,y=value_y)

#Presentation text
    label_text_box = ttk.Label(self,text="Welcome to image\nclassification app\nusing Machine\nLearning and Deep\nLearning model!\n\nSteps:\n1:Select or drop an\nimage\n2:Select a model\n3:Apply model\n4:Get Results!")
    label_text_box.place(x=5,y=10)

# create sidebar frame with widgets
    
    
    #Functions for other button
    @staticmethod
    def adapt_text_model(radio_val):
      if radio_val == 0:
          return "KNN.sav"
      elif radio_val == 1:
          return "SVC.sav"
      elif radio_val == 2:
          return "LinearSVC.sav"
      elif radio_val == 3:
          return "LogReg.sav"
      elif radio_val == 4:
          return "ML-Perceptron.keras"
      elif radio_val == 5:
          return "CNN.keras"

    @staticmethod
    def create_pred_text(prediction):
        if prediction[0] == 0:
            prediction_text = "T-Shirt"
            prediction_proba = f"Probabilité associé: {round(float(prediction[1][0]),2)*100}%"
        elif prediction[0] == 1:
            prediction_text = "Pantalon"
            prediction_proba = f"Probabilité associé: {round(float(prediction[1][1]),2)*100}%"
        elif prediction[0] == 2:
            prediction_text = "Pullover"
            prediction_proba = f"Probabilité associé: {round(float(prediction[1][2]),2)*100}%"
        elif prediction[0] == 3:
            prediction_text = "Jupe"
            prediction_proba = f"Probabilité associé: {round(float(prediction[1][3]),2)*100}%"
        elif prediction[0] == 4:
            prediction_text = "Manteau"
            prediction_proba = f"Probabilité associé: {round(float(prediction[1][4]),2)*100}%"
        elif prediction[0] == 5:
            prediction_text = "Sandales"
            prediction_proba = f"Probabilité associé: {round(float(prediction[1][5]),2)*100}%"
        elif prediction[0] == 6:
            prediction_text = "Chemise"
            prediction_proba = f"Probabilité associé: {round(float(prediction[1][6]),2)*100}%"
        elif prediction[0] == 7:
            prediction_text = "Sneakers"
            prediction_proba = f"Probabilité associé: {round(float(prediction[1][7]),2)*100}%"
        elif prediction[0] == 8:
            prediction_text = "Sac"
            prediction_proba = f"Probabilité associé: {round(float(prediction[1][8]),2)*100}%"
        elif prediction[0] == 9:
            prediction_text = "Bottines"
            prediction_proba = f"Probabilité associé: {round(float(prediction[1][9]),2)*100}%"
        return (prediction_text,prediction_proba)

    @staticmethod
    def select_file():
            filetypes = (('JPG files', '*.jpg'),('PNG files', '*.png'),('All files', '*.*'))
            filepath = fd.askopenfilename(title='Open image',initialdir='/',filetypes=filetypes)
            self.filepathimage = f"{filepath}"
            image = Image.open(f"{filepath}")
            reside_image = image.resize((100, 100))
            image = ImageTk.PhotoImage(reside_image,size=(100,100))
            tk.Label(self, image=image,text="").place(x=200,y=20)
            self.mainloop()
    
    @staticmethod
    def dnd_file(event):
            testvariable.set(event.data)
            self.filepathimage = f"{str(event.data)}"
            image = Image.open(self.filepathimage)
            # resize image
            reside_image = image.resize((100, 100))
            image = ImageTk.PhotoImage(reside_image,size=(100,100))
            tk.Label(self, image=image,text="").place(x=200,y=20)
            self.mainloop()

    @staticmethod
    def call_model():
        model_text = adapt_text_model(self.radio_var.get())
        prediction = Transform_prediction.prediction(model_path=model_text, image_to_predict = self.filepathimage)
        pred_text = create_pred_text(prediction)
        label1.config(text= pred_text[0])
        label2.config(text= pred_text[1])
        

    testvariable = tk.StringVar()
    label3_frame = tk.Label(self,justify="center",text="Drag'n Drop your image below")
    label3_frame.place(x=175,y=200)
    entrybox = ttk.Entry(master=self, textvar=testvariable,width=30)
    entrybox.place(x=175,y=225)
    entrybox.drop_target_register(DND_FILES)
    entrybox.dnd_bind('<<Drop>>',dnd_file)

    #Apply model button
    self.myButton = tk.Button(self, text="Apply Model",command=call_model).place(x=400,y=275)
    
    #Open Image Button Code
    self.open_button = tk.Button(self,text='Select an Image',command=select_file).place(x=75,y=222)


if __name__ == "__main__":
  app = App()
  app.mainloop()
