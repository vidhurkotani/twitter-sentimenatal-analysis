#!/usr/bin/env python
# coding: utf-8

# In[3]:

import pickle
from tkinter import *
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
with open('tok.pkl','rb') as file:
    tokenize=pickle.load(file)
model = load_model('project.h5')
top = Tk()
top.geometry("550x300+300+150")
top.resizable(width=True, height=True)

L1 = Label(top, text="Enter Text For Prediction")
L1.pack()
E1 = Entry(top, bd =5)
E1.pack()
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
#max_words=1500
#tokenize=Tokenizer(num_words=max_words,char_level=False)
def predict1():
    print("Prediction on progress..")
    entered_input=E1.get()
    print("Entered Input",entered_input)
    entered_input=tokenize.texts_to_matrix([entered_input])
    print(entered_input)
    y_pred=model.predict_classes(entered_input)
    print(y_pred)
    if y_pred[0]==0:
        text="The input is a NEGATIVE sentiment"
        
    else:
        text="The input is a POSITIVE sentiment"

    L2 = Label(top, text="Prediction: "+text)
    L2.pack()


B = Button(top, text ="Predict", command = predict1)
B.pack(pady=10)

top.mainloop()


# In[ ]:





# In[ ]:




