import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

class App(tk.Tk):
  def __init__(self):
    super().__init__()

    # configure the root window
    self.title('GUI')
    self.geometry('400x400')

    self.frame = tk.Frame(self)

    self.str_menu = tk.StringVar()
    self.str_menu.set("KNN")
    tk.OptionMenu(self.frame, self.str_menu, "KNN", "SVC", "LinearSVC", "Logistic Regression").grid(column=1,row=0)

    #Exit Button
    tk.Button(self.frame,text='Exit',command=lambda: self.quit()).grid(column=0,row=0)

    #Functions for other button
    @staticmethod
    def call_model():
        tk.Label(self.frame, text=self.str_menu.get()).grid(column=1,row=5)

    @staticmethod
    def select_file():
            filetypes = (('JPG files', '*.jpg'),('PNG files', '*.png'),('All files', '*.*'))
            filename = fd.askopenfilename(title='Open a file',initialdir='/',filetypes=filetypes)
            showinfo(title='Selected File',message=filename)
    
    #Select model button
    self.myButton = tk.Button(self.frame, text="Apply Model",command=call_model).grid(column=2,row=0)
    
    #Open Image Button Code
    self.open_button = tk.Button(self.frame,text='Select an Image',command=select_file).grid(column=3,row=0)

    self.frame.pack()

if __name__ == "__main__":
  app = App()
  app.mainloop()
