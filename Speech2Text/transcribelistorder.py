#!/usr/bin/python
# -*- coding: UTF-8 -*-
import csv
import os
datas=csv.reader(open('./data/After_transcribelist.csv'),delimiter=',')
sortedlist=sorted(datas,key=lambda x: (int(x[0])))
with open("./data/After_transcribelist.csv","w") as f:
	writer=csv.writer(f)
	for row in sortedlist:
		writer.writerow(row)
		print(row)
f.close()

			


	
