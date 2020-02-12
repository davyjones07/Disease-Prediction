from tkinter import Tk,StringVar,Label,Entry,OptionMenu,Button,Text,END
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


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





# ------------------------------------------------------------------------------------------------------
import time

def DecisionTree():

    if(Symptom1.get()=='Select Symptom'):
        t1.delete("1.0", END)
        t1.insert(END, 'Select atleast 1 Symptom')
    else:
        
        
        clf3 = tree.DecisionTreeClassifier()   
        tm0=time.time()
        clf3 = clf3.fit(X,y)
        print( "training time:", round(time.time()-tm0, 3), "s" )
        tm1=time.time()
        from sklearn.metrics import accuracy_score
        y_pred=clf3.predict(X_test)        
        print( "predict time:", round(time.time()-tm1, 3), "s")
        print(accuracy_score(y_test, y_pred))
        
        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    
        for k in range(0,len(symptoms_list)):
            for z in psymptoms:
                if(z==symptoms_list[k]):
                    symptoms_count_list[k]=1
    
        inputtest = [symptoms_count_list]
        predict = clf3.predict(inputtest)
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
    
    
    

def randomforest():
    
    if(Symptom1.get()=='Select Symptom'):
        t1.delete("1.0", END)
        t1.insert(END, 'Select atleast 1 Symptom')
    else:
        
            
        clf4 = RandomForestClassifier()
        clf4 = clf4.fit(X,np.ravel(y))
    
    #    from sklearn.metrics import accuracy_score
    #    y_pred=clf4.predict(X_test)
    #    print(accuracy_score(y_test, y_pred))
    
    
    
        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    
        for k in range(0,len(symptoms_list)):
            for z in psymptoms:
                if(z==symptoms_list[k]):
                    symptoms_count_list[k]=1
    
        inputtest = [symptoms_count_list]
        predict = clf4.predict(inputtest)
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
    

def NaiveBayes():
    
    if(Symptom1.get()=='Select Symptom'):
        t1.delete("1.0", END)
        t1.insert(END, 'Select atleast 1 Symptom')
    else:
    
        gnb = GaussianNB()
        gnb=gnb.fit(X,np.ravel(y))
    
    
    #    from sklearn.metrics import accuracy_score
    #    y_pred=gnb.predict(X_test)
    #    print(accuracy_score(y_test, y_pred))
    
    
    
        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
        for k in range(0,len(symptoms_list)):
            for z in psymptoms:
                if(z==symptoms_list[k]):
                    symptoms_count_list[k]=1
    
        inputtest = [symptoms_count_list]
        predict = gnb.predict(inputtest)
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
    
    
    

#--------------------------------------------------GRAPHICAL USER INTERFACE-------------------------------

root = Tk()
root.configure(background='black')
root.geometry('1200x550')
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

# Heading
w2 = Label(root, text="A GENERAL DISEASE PREDICTOR", fg="white", bg="black")
w2.config(font=("Arial", 20))
w2.grid(row=1, column=0, columnspan=2,pady=50,padx=350)

# labels
NameLb = Label(root, text="Name of the Patient", fg="yellow", bg="black")
NameLb.grid(row=6, column=0, pady=15)


S1Lb = Label(root, text="Symptom 1", fg="yellow", bg="black")
S1Lb.grid(row=7, column=0, pady=10)

S2Lb = Label(root, text="Symptom 2", fg="yellow", bg="black")
S2Lb.grid(row=8, column=0, pady=10)

S3Lb = Label(root, text="Symptom 3", fg="yellow", bg="black")
S3Lb.grid(row=9, column=0, pady=10)

S4Lb = Label(root, text="Symptom 4", fg="yellow", bg="black")
S4Lb.grid(row=10, column=0, pady=10)

S5Lb = Label(root, text="Symptom 5", fg="yellow", bg="black")
S5Lb.grid(row=11, column=0, pady=10)


lrLb = Label(root, text="Decision Tree", fg="white", bg="red")
lrLb.grid(row=15, column=0, pady=10)

destreeLb = Label(root, text="Random Forest", fg="white", bg="red")
destreeLb.grid(row=17, column=0, pady=10)

ranfLb = Label(root, text="Naive Bayes", fg="white", bg="red")
ranfLb.grid(row=19, column=0, pady=10)

# entries
OPTIONS = sorted(symptoms_list)

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)
NameEn.config(width=56)


S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=7, column=1)
S1En.config(width=50)


S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=8, column=1)
S2En.config(width=50)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=9, column=1)
S3En.config(width=50)


S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=10, column=1)
S4En.config(width=50)


S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=11, column=1)
S5En.config(width=50)



t1 = Text(root, height=1, width=40,bg="orange",fg="black")
t1.grid(row=15, column=1, padx=10)

t2 = Text(root, height=1, width=40,bg="orange",fg="black")
t2.grid(row=17, column=1 , padx=10)

t3 = Text(root, height=1, width=40,bg="orange",fg="black")
t3.grid(row=19, column=1 , padx=10)




dst = Button(root, text="Decision Tree", command=DecisionTree,bg="green",fg="yellow")
dst.grid(sticky="e",row=15, column=1)

rnf = Button(root, text="Random Forest", command=randomforest,bg="green",fg="yellow")
rnf.grid(sticky="e",row=17,column=1)

lr = Button(root, text="Naive Bayes", command=NaiveBayes,bg="green",fg="yellow")
lr.grid(sticky="e",row=19, column=1)


root.mainloop()



#------------PLOTS---------------------#
#
#import seaborn as sns
#sns.set(style="ticks", color_codes=True)
#
#g = sns.pairplot(df)
#
#
#
#############    COUNT PLOT OF DISEASE    ########################
#from matplotlib import pyplot as plt
#import seaborn as sns
#
#
#dfd=pd.read_csv("Training.csv")
#ddd=dfd.iloc[:,-1]
#
#a4_dims = (5, 20)
#
#fig, ax = plt.subplots(figsize=a4_dims)
#ax = sns.countplot(y=ddd, data=df)
#ax.set(title = 'COUNT PLOT OF DISEASES', xlabel='COUNT OF DISEASES', ylabel='DIFFERENT TYPES OF DISEASES')
#
#
#
#
##################   COUNT PLOT SYMPTOMS        ##############################
#
#
#dfsy=dfd.iloc[:,:-1]
#
#a4_dims = (5, 20)
#
##fig, ax = pyplot.subplots(figsize=a4_dims)
#ax = sns.countplot(y=np.ravel(df[symptoms_list]), data=df)
#ax.set(title = 'COUNT PLOT OF SYMPTOMS', xlabel='COUNT OF SYMPTOMS', ylabel='DIFFERENT TYPES OF DISEASES')
#
#
#
#
#
#
#g = sns.pairplot(df[symptoms_list], hue="species")
#
#
#
#
#
#
#
#
#df[symptoms_list].hist(bins=15, figsize=(100, 100));
#
#ax = sns.boxplot(data=df, orient="h", palette="Set2")