To run need : mathplotlib,
seaborn,
pandas,
numpy,
scipy,
scikit-learn,
keras,
tensorflow,
skimage,
tkinter,
customtkinter,
tkinterdnd2

You must modify your customtkinter module in order to make it compatible with tkinterdnd2, to do so follow those steps:

1. Download ctk.py from this git-hub
2. open path C:\Users\username\anaconda3\Lib\site-packages\customtkinter\windows (don't forget to change username in the path) and change ctk.py by the one you download
3. End by running GUI_OOP.py, it should now work without error message

I am working on an update to go back to tkinter so we can create an exe (wich curently isn't possible for compatibility problems) and also reoptimise models
