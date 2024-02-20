import tkinter as tk
import tkinter.messagebox
import customtkinter as ctk
import Transform_prediction
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from PIL import ImageTk,Image

ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
ctk.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green

class App(ctk.CTk):
  def __init__(self):
    super().__init__()

    # configure the root window
    self.title('GUI TER')
    self.geometry('600x600')
    self.filepathimage = ""
    self.radio_var = tkinter.IntVar(value=0)
    # configure grid layout (4x4)
    self.grid_columnconfigure(1, weight=1)
    self.grid_columnconfigure((2, 3), weight=0)
    self.grid_rowconfigure((0, 1, 2), weight=1)

    self.frame = ctk.CTkFrame(self,width=200)
    self.frame.grid(row=0, column=1, columnspan=2, padx=(0, 0), pady=(0, 0))
    label1_frame = ctk.CTkLabel(self.frame, text="")
    label1_frame.grid(column=1,row=1)
    label2_frame = ctk.CTkLabel(self.frame, text="")
    label2_frame.grid(column=1,row=2)

# create radiobutton frame
    self.radiobutton_frame = ctk.CTkFrame(self)
    self.radiobutton_frame.grid(row=0, column=3, padx=(0, 0), pady=(0, 0), sticky="ne")
    self.label_radio_group = ctk.CTkLabel(master=self.radiobutton_frame, text="Modèles:")
    self.label_radio_group.grid(row=0, column=5, columnspan=1, sticky="")
    self.radio_button_1 = ctk.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=0, text="KNN")
    self.radio_button_1.grid(row=1, column=5, pady=10, padx=10, sticky="w")
    self.radio_button_2 = ctk.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=1, text="Support Vector Machine")
    self.radio_button_2.grid(row=2, column=5, pady=10, padx=10, sticky="w")
    self.radio_button_3 = ctk.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=2, text="Linear Support Vector Machine")
    self.radio_button_3.grid(row=3, column=5, pady=10, padx=10, sticky="w")
    self.radio_button_3 = ctk.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=3, text="Logistic Regression")
    self.radio_button_3.grid(row=4, column=5, pady=10, padx=10, sticky="w")
    self.radio_button_4 = ctk.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=4, text="Multi-Layers Perceptron")
    self.radio_button_4.grid(row=5, column=5, pady=10, padx=10, sticky="w")
    self.radio_button_4 = ctk.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=5, text="Convolution Neural Network")
    self.radio_button_4.grid(row=6, column=5, pady=10, padx=10, sticky="w")

# create sidebar frame with widgets
    self.sidebar_frame = ctk.CTkFrame(self)
    self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
    
    #Exit Button
    ctk.CTkButton(self.sidebar_frame,text='Exit',command=lambda: self.quit()).grid(row=0,column=0)

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
          return "Sequential.keras"
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
    def call_model():
        model_text = adapt_text_model(self.radio_var.get())
        prediction = Transform_prediction.prediction(model_path=model_text, image_to_predict = self.filepathimage)
        pred_text = create_pred_text(prediction)
        label1_frame.configure(text=pred_text[0])
        label2_frame.configure(text=pred_text[1])
        

    @staticmethod
    def select_file():
            filetypes = (('JPG files', '*.jpg'),('PNG files', '*.png'),('All files', '*.*'))
            filepath = fd.askopenfilename(title='Open image',initialdir='/',filetypes=filetypes)
            self.filepathimage = f"{filepath}"
            image = Image.open(f"{filepath}")
            image = image.resize((100,100))
            image = ImageTk.PhotoImage(image)
            ctk.CTkLabel(self.frame, image=image,text="").grid(column=1,row=0)
            self.mainloop()
    
    #Apply model button
    self.myButton = ctk.CTkButton(self.radiobutton_frame, text="Apply Model",command=call_model).grid(column=5,row=7, pady=10, padx=10)
    
    #Open Image Button Code
    self.open_button = ctk.CTkButton(self,text='Select an Image',command=select_file).grid(column=1,row=0,sticky='n')

if __name__ == "__main__":
  app = App()
  app.mainloop()
