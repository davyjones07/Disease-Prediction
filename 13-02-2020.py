from tkinter import Tk,StringVar,Label,Entry,OptionMenu,Button,Text,END
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


symptoms_list=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']



symptoms_count_list=[]

for x in range(0,len(symptoms_list)):
    symptoms_count_list.append(0)



# TRAINING DATA df -------------------------------------------------------------------------------------


df=pd.read_csv("Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)



X= df[symptoms_list]

y = df[["prognosis"]]

np.ravel(y)



# TESTING DATA tr --------------------------------------------------------------------------------




tr=pd.read_csv("Testing.csv")


tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[symptoms_list]

y_test = tr[["prognosis"]]

np.ravel(y_test)


oplist=df.columns


# ------------------------------------------------------------------------------------------------------


def DecisionTree():

    if(Symptom1.get()=='Select Symptom'):
        t1.delete("1.0", END)
        t1.insert(END, 'Select atleast 1 Symptom')
    else:
        

        dec_tree_clf = tree.DecisionTreeClassifier()   
        dec_tree_clf = dec_tree_clf.fit(X,y)
        
        dec_tree_acc=dec_tree_clf.score(X_test,y_test)
        print('DECISION TREE SCORE :',dec_tree_acc)        
        
        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    
        for k in range(0,len(symptoms_list)):
            for z in psymptoms:
                if(z==symptoms_list[k]):
                    symptoms_count_list[k]=1
    
        inputtest = [symptoms_count_list]
        predict = dec_tree_clf.predict(inputtest)
        predicted=predict[0]
    
        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break
    
        if (h=='yes'):
            t1.delete("1.0", END)
            t1.insert(END, disease[a])
        else:
            t1.delete("1.0", END)
            t1.insert(END, "Not Found")
    
    
    

def NaiveBayes():
    
    if(Symptom1.get()=='Select Symptom'):
        t2.delete("1.0", END)
        t2.insert(END, 'Select atleast 1 Symptom')
    else:
    
        NB_clf = GaussianNB()
        NB_clf=NB_clf.fit(X,np.ravel(y))
    
    
  
        NB_acc=NB_clf.score(X_test,y_test)
        print('NAIVE BAYES ACCURACY SCORE :',NB_acc)
    
    
    
        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
        for k in range(0,len(symptoms_list)):
            for z in psymptoms:
                if(z==symptoms_list[k]):
                    symptoms_count_list[k]=1
    
        inputtest = [symptoms_count_list]
        predict = NB_clf.predict(inputtest)
        predicted=predict[0]
    
        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break
    
        if (h=='yes'):
            t2.delete("1.0", END)
            t2.insert(END, disease[a])
        else:
            t2.delete("1.0", END)
            t2.insert(END, "Not Found")
    

def knn():
    
    if(Symptom1.get()=='Select Symptom'):
        t3.delete("1.0", END)
        t3.insert(END, 'Select atleast 1 Symptom')
    else:
    
        knn_clf = KNeighborsClassifier(n_neighbors=3)
        knn_clf.fit(X, np.ravel(y))
        
        knn_acc=knn_clf.score(X_test,y_test)
        print('KNN ACCURACY SCORE :',knn_acc)
    
    
        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
        for k in range(0,len(symptoms_list)):
            for z in psymptoms:
                if(z==symptoms_list[k]):
                    symptoms_count_list[k]=1
    
        inputtest = [symptoms_count_list]
        predict = knn_clf.predict(inputtest)
        predicted=predict[0]
    
        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break
    
        if (h=='yes'):
            t3.delete("1.0", END)
            t3.insert(END, disease[a])
        else:
            t3.delete("1.0", END)
            t3.insert(END, "Not Found")
            
            
            
''''--------------------------------------------------GRAPHICAL USER INTERFACE-------------------------------'''

    
    
root = Tk()


def close_window(): 
    root.destroy()


#root.attributes("-fullscreen", True)
root.configure(background='black')
root.title('MACHINE LEARNING DISEASE PREDICTOR.')

# entry variables
Symptom1 = StringVar()
Symptom1.set('Select Symptom')
Symptom2 = StringVar()
Symptom2.set('Select Symptom')
Symptom3 = StringVar()
Symptom3.set('Select Symptom')
Symptom4 = StringVar()
Symptom4.set('Select Symptom')
Symptom5 = StringVar()
Symptom5.set('Select Symptom')
Name = StringVar()




'''   MAIN HEAGING   '''

w2 = Label(root, text="A GENERAL DISEASE PREDICTOR MACHINE.", fg="white", bg="black", font='Arial 20 bold')
w2.place(x=350,y=50)



''' NAME OF PATIENT LABEL AND TEXTBOX '''

NameLb = Label(root, text="Name of the Patient", fg="yellow", bg="black", font='Arial 13')
NameLb.place(x=350,y=250)

NameEn = Entry(root, textvariable=Name,width=57)
NameEn.place(x=600,y=250)


''' SYMPTOM LABELS AND OPTION BUTTONS '''

''' LABELS'''

S1Lb = Label(root, text="Symptom 1", fg="yellow", bg="black", font='Arial 13')
S1Lb.place(x=350,y=300)

S2Lb = Label(root, text="Symptom 2", fg="yellow", bg="black", font='Arial 13')
S2Lb.place(x=350,y=350)

S3Lb = Label(root, text="Symptom 3", fg="yellow", bg="black", font='Arial 13')
S3Lb.place(x=350,y=400)

S4Lb = Label(root, text="Symptom 4", fg="yellow", bg="black", font='Arial 13')
S4Lb.place(x=350,y=450)

S5Lb = Label(root, text="Symptom 5", fg="yellow", bg="black", font='Arial 13')
S5Lb.place(x=350,y=500)


'''OPTIONS'''

OPTIONS = sorted(oplist)


S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.place(x=600,y=300)
S1En.config(width=50)


S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.place(x=600,y=350)
S2En.config(width=50)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.place(x=600,y=400)
S3En.config(width=50)


S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.place(x=600,y=450)
S4En.config(width=50)


S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.place(x=600,y=500)
S5En.config(width=50)


''' ALGORITHM NAME LABELS AND TEXTBOX AND BUTTONS'''

'''LABELS'''
lrLb = Label(root, text="Decision Tree", fg="white", bg="red", font='Arial 13')
lrLb.place(x=350,y=600)

destreeLb = Label(root, text="Naive Bayes", fg="white", bg="red", font='Arial 13')
destreeLb.place(x=350,y=650)

ranfLb = Label(root, text="KNeighborsClassifier", fg="white", bg="red", font='Arial 13')
ranfLb.place(x=350,y=700)


'''TEXTBOXES'''

t1 = Text(root, height=1, width=40,bg="orange",fg="black")
t1.place(x=600,y=600)

t2 = Text(root, height=1, width=40,bg="orange",fg="black")
t2.place(x=600,y=650)

t3 = Text(root, height=1, width=40,bg="orange",fg="black")
t3.place(x=600,y=700)


''' BUTTONS   '''

dst = Button(root, text="Decision Tree", command=DecisionTree,bg="green",fg="yellow", font='Arial 10', width=20)
dst.place(x=980,y=600)

rnf = Button(root, text="Naive Bayes", command=NaiveBayes,bg="green",fg="yellow", font='Arial 10', width=20)
rnf.place(x=980,y=650)

lr = Button(root, text="K N N", command=knn,bg="green",fg="yellow", font='Arial 10', width=20)
lr.place(x=980,y=700)



'''CLOSE BUTTON'''

lr = Button(root, text="CLOSE", command=close_window,bg="red",fg="yellow", font='Arial 10')
lr.place(x=1200,y=950)


root.mainloop()