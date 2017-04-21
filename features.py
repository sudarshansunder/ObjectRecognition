import numpy as np
from PIL import Image
import math

#Histogram of oriented gradients

def hist_img(img, x, y):
	hog_points = [0, 20, 40, 60, 80, 100, 120, 140, 160]
	hogr = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	hogg = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	hogb = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	feature_vector = []
	h, w = 8, 8
	for i in xrange(h):
		for j in xrange(w):
			pixel = list(img[i,j])
			pu = list(img[abs(i-1), j]) 
			pd = list(img[(i+1)%h, j]) 
			pl = list(img[i, abs(j-1)])
			pr = list(img[i, (j+1)%w]) 
	
			if i == 0:		
				pu = pd
			if i == h:
				pd = pu
			if j == 0:
				pl = pr
			if j == w:
				pr = pl
				
			dxr = (pr[0] - pl[0])
			dyr = (pu[0] - pd[0])
			magGradr = math.sqrt(pow(dxr, 2) + pow(dyr, 2))
			if dxr != 0:
				degr = math.atan(dyr/dxr)
			else:
				if dyr >= 0:
					degr = 3.14/2
				else :
					degr = - 3.14/2
			degr *= 57.2958
			if degr < 0:
				degr += 180
						
			tr = int(math.floor(degr/20))
			dr =(20 -( degr - hog_points[tr])) /20
			hogr[tr] += dr * magGradr
			hogr[(tr+1)%9] += abs(1-dr) * magGradr

		
			dxg = (pr[1] - pl[1])
			dyg = (pu[1] - pd[1])
		
			magGradg = math.sqrt(pow(dxg, 2) + pow(dyg, 2))
			if dxg != 0:
				degg = math.atan(dyg/dxg)
			else:
				if dyg >= 0:
					degg = 3.14/2
				else :
					degg = - 3.14/2
			degg *= 57.2958
			if degg < 0:
				degg += 180
			
			tg = int(math.floor(degg/20))
			dg = (20 -(degg - hog_points[tg])) /20
			hogg[tg] += dg * magGradg
			hogg[(tg+1)%9] += abs(1-dg) * magGradg

			dxb = (pr[2] - pl[2])
			dyb = (pu[2] - pd[2])

			magGradb = math.sqrt(pow(dxb, 2) + pow(dyb, 2))
			if dxb != 0:
				degb = math.atan(dyb/dxb)
			else:
				if dyb >= 0:
					degb = 3.14/2
				else:
					degb = - 3.14/2
			degb *= 57.2958
		
			if degb < 0:
				degb += 180
			tb = int(math.floor(degb/20))
			db = (20 -(degb - hog_points[tb])) /20
			hogb[tb] += db * magGradb
			hogb[(tb+1)%9] += abs(1-db) * magGradb
			

	for z in xrange(9):
		feature_vector.append(math.sqrt(pow(hogr[z], 2) + pow(hogr[z], 2) +pow(hogr[z], 2)))
	return feature_vector

def getFeatures(img_name):
	img_n = Image.open(img_name)
	img = img_n.load()
	h1, w1 = img_n.size
	h1 = h1/8
	w1= w1/8
	vector = []
	feat = []
	for i in xrange(h1):
		for j in xrange(w1):
			i_n =  Image.new( 'RGB', (8, 8), "black")
			pixels = i_n.load()
			for x in xrange(8):
				for y in xrange(8):
					pixels[x,y] = img[(i*8)+x,(j*8)+y] 	
			feat = hist_img(pixels, i, j)
			for x in xrange(len(feat)):
				vector.append(feat[x])	
	return vector
