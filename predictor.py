import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk import *
import csv
txt=[]
lab=[]
with open('[your classifier filename]') as csvfile:
	reader=csv.DictReader(csvfile)
	for row in reader:	
		t=str(row['TEXT']),str(row['LABEL'])
		txt.append(t[0])
		lab.append(t[1])

print (len(txt))
print (len(lab))


count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(txt)
transformer = TfidfTransformer(use_idf=False).fit(train_counts)
train_tf = transformer.transform(train_counts)
clf = MultinomialNB().fit(train_tf, lab)

docs_new =[]
with open('[your texts you want to analyze]') as csvfile:
	reader=csv.DictReader(csvfile)
	for row in reader:	
		t=str(row['TEXT'])
		docs_new.append(t)


new_counts = count_vect.transform(docs_new)
new_tf = transformer.transform(new_counts)
predicted = clf.predict(new_tf)


label=[]
y=1

for i,x in zip(docs_new,predicted):
##USE THIS LINES IF YOU WANT A HUMAN SUPERVISION#####
	print (i,'LABEL:',x)
	rep=input('OK?')
	if rep=='':
		wr=i,x
		label.append(wr)
	else:
		wr=i,'1'
		label.append(wr)
	print (len(docs_new)-y,'TO GO')
	y+=1
#########################################################
##USE THIS LINES IF YOU DON'T WANT A HUMAN SUPERVISION###
  wr=i,x
	label.append(wr)
##########################################################

with open('[where you want to save the results of this A.I.]','a') as csvfile:
	fieldnames = ['TEXT','LABEL']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	for i in label:
		writer.writerow({'TEXT':i[0],'LABEL':i[1]})
