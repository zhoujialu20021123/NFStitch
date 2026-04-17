import numpy as np
import math
import computer_vision_utils as cv_util
import multiprocessing
import random
import sys
import datetime
from scipy.sparse.linalg import lsmr
from scipy.optimize import lsq_linear
from scipy.optimize import leastsq,least_squares

def report_time(start,end):
	print(':: Optimization\n\tStart: {0}\n\tEnd: {1}\n\tTotal running time: {2}.'.format(start,end,end-start))

def get_translation_in_GPS_coordinate_system(T,Coord_TMP,W,H):

	c1 = [0,0,1]
	
	c1 = T.dot(c1).astype(int)

	diff_x = -c1[0]
	diff_y = -c1[1]

	Patch_Size_GPS = (Coord_TMP['UR']['lon'] - Coord_TMP['UL']['lon'],Coord_TMP['UL']['lat'] - Coord_TMP['LL']['lat'])
	Patch_Size = (W,H)

	gps_scale_x = -(Patch_Size_GPS[0])/(Patch_Size[0])
	gps_scale_y = (Patch_Size_GPS[1])/(Patch_Size[1])

	diff_x = diff_x*gps_scale_x
	diff_y = diff_y*gps_scale_y

	return (diff_x,diff_y)

class Graph:

	def __init__(self,images,pairwise_homography_dict,image_name_to_index_dict,ref_name):

		self.images = images

		self.images_dict = {}

		for img in self.images:
			self.images_dict[img.name] = img

		self.pairwise_homography_dict = pairwise_homography_dict
		self.image_name_to_index_dict = image_name_to_index_dict
		self.reference_image = self.images_dict[ref_name]

		self.image_index_to_name_dict = {}

		for name in self.image_name_to_index_dict:
			self.image_index_to_name_dict[self.image_name_to_index_dict[name]] = name

		self.edge_matrix = np.zeros((len(self.images),len(self.images)))

		for img_name in self.pairwise_homography_dict:
			for neighbor_name in self.pairwise_homography_dict[img_name]:

				i = self.image_name_to_index_dict[img_name]
				j = self.image_name_to_index_dict[neighbor_name]

				self.edge_matrix[i][j] = self.pairwise_homography_dict[img_name][neighbor_name][2]

				if neighbor_name in self.pairwise_homography_dict and img_name in self.pairwise_homography_dict[neighbor_name]:
					if self.edge_matrix[j][i]>self.edge_matrix[i][j]:
						self.edge_matrix[i][j] = self.edge_matrix[j][i]
					else:
						self.edge_matrix[j][i] = self.edge_matrix[i][j]

		self.node_count = len(self.images)

		self.MST = self.generate_MST_prim(self.image_name_to_index_dict[self.reference_image.name])

		# self.absolute_homography_dict = self.get_absolute_homographies()

	def find_min_key(self,keys,mstSet):
		min_value = sys.maxsize
		
		for v in range(self.node_count): 
			if keys[v] < min_value and mstSet[v] == False: 
				min_value = keys[v] 
				min_index = v 
		
		return min_index 

	def generate_MST_prim(self,starting_vertex):

		keys = [sys.maxsize]*self.node_count
		parents = [None]*self.node_count
		mstSet = [False]*self.node_count

		keys[starting_vertex] = 0
		parents[starting_vertex] = -1

		for count in range(self.node_count):
			u = self.find_min_key(keys,mstSet)
			mstSet[u] = True

			for v in range(self.node_count):
				
				if self.edge_matrix[u][v] > 0 and mstSet[v] == False and keys[v] > self.edge_matrix[u][v]:
					keys[v] = self.edge_matrix[u][v]
					parents[v] = u

		
		new_edges = np.zeros((self.node_count,self.node_count))

		queue_traverse = []
		
		for v,p in enumerate(parents):
			if p == -1:
				queue_traverse = [v]
				break
		
		while len(queue_traverse) > 0:
			u = queue_traverse.pop()

			for v,p in enumerate(parents):
				if p == u:
					queue_traverse = [v] + queue_traverse

					new_edges[p][v] = self.edge_matrix[p][v]
					new_edges[v][p] = self.edge_matrix[v][p]

		# g = Graph(self.images,self.image_name_to_index_dict,new_edges)

		# return g
		return new_edges

	def get_absolute_homographies(self):
		
		absolute_homography_dict = {}

		queue_traverse = [self.reference_image.name]
		
		absolute_homography_dict[self.reference_image.name] = np.eye(3)

		while len(queue_traverse) > 0:
			u = queue_traverse.pop()

			for v,edge in enumerate(self.MST[self.image_name_to_index_dict[u]]):
				
				v_name = self.image_index_to_name_dict[v]

				if v_name in absolute_homography_dict:
					continue

				if edge != 0:
					
					absolute_u = absolute_homography_dict[u]
					H = np.matmul(absolute_u,self.pairwise_homography_dict[u][v_name][0])
					absolute_homography_dict[v_name] = H

					queue_traverse = [v_name] + queue_traverse

		return absolute_homography_dict

	def get_coords_from_absolute_homographies(self,x,y):

		absolute_homography_dict= self.get_absolute_homographies()

		image_corners_dict = {}
		
		UL_ref = [0,0,1]
		UR_ref = [x,0,1]
		LR_ref = [x,y,1]
		LL_ref = [0,y,1]
		
		for img_name in absolute_homography_dict:
			
			H = absolute_homography_dict[img_name]

			UL = np.matmul(H,UL_ref)
			UL = UL/UL[2]
			UL = UL[:2]

			UR = np.matmul(H,UR_ref)
			UR = UR/UR[2]
			UR = UR[:2]

			LR = np.matmul(H,LR_ref)
			LR = LR/LR[2]
			LR = LR[:2]

			LL = np.matmul(H,LL_ref)
			LL = LL/LL[2]
			LL = LL[:2]

			image_corners_dict[img_name] = {'UL':UL,'UR':UR,'LR':LR,'LL':LL}

		return image_corners_dict

class ReprojectionMinimization:

	def __init__(self,images,pairwise_trans,coefs,w,h,ref_name,max_mat,k,neighbor=None,scale=1,keypoint=1, gridSize = (3, 3), working = None, no_gps_distance = False, X_run = 0, Y_run = 0):

		self.images = images
		self.images_dict = {}#图片名称和图片对应
		self.n_to_i_dict = {}#图片名称和索引对应

		for i,img in enumerate(self.images):
			self.images_dict[img.name] = img
			self.n_to_i_dict[img.name] = i

		self.num_images = len(self.images)

		self.pairwise_transformations = pairwise_trans
		self.image_reference_name = ref_name
		self.width = w
		self.height = h
		self.keypoint = keypoint
		self.grid_size = gridSize
		self.working = working
		self.no_gps_distance = no_gps_distance
		self.X_run = X_run
		self.Y_run = Y_run

		self.max_matches_to_use = max_mat#最大内点个数
		self.coefs = coefs#权重
		self.scale_images = scale
		self.cross_validation_k = k#-1
		if neighbor is not None:
			self.ori_neighbor = neighbor

	def Estimate_GSD(self):
		Neighbor_image = None
		min_distance_to_center = sys.maxsize
		for img in self.images:
			if img.lat==None or img.lon==None or img.name ==self.image_reference_name:
				continue
			d = cv_util.get_gps_distance(
				self.images_dict[self.image_reference_name].lat,self.images_dict[self.image_reference_name].lon, img.lat, img.lon
			)
			if d < min_distance_to_center:
				min_distance_to_center = d
				Neighbor_image = img
		print(self.image_reference_name,Neighbor_image.name)
		H = self.pairwise_transformations[self.image_reference_name][Neighbor_image.name][0]
		
		width = self.width/2
		height = self.height/2
		PixX =  width- (H[0][0]*width+H[0][1]*height+H[0][2])
		PixY = height - (H[1][0]*width+H[1][1]*height+H[1][2])
		PixZ = math.sqrt(PixX*PixX+PixY*PixY)
		GPSZ = cv_util.get_gps_distance(Neighbor_image.lat,Neighbor_image.lon,self.images_dict[self.image_reference_name].lat,self.images_dict[self.image_reference_name].lon)
		GSD = GPSZ/PixZ
		print(PixX,PixY,PixZ,GPSZ,GSD,cv_util.get_GSD(Neighbor_image.Pixsize, Neighbor_image.FocalLength, Neighbor_image.height, self.scale_images))#像素大小（像元微米）、相机焦距（毫米）、飞行高度（米）
		return GSD
		

	
	def Get_Pix_DistanceAndAngle(self, img_A_name,img_B_name,EstimatedGSD=None):
		img_A = self.images_dict[img_A_name]
		img_B = self.images_dict[img_B_name]
		# theta = img_A.Yaw-img_B.Yaw
		theta = None
		# GSD表示图像中一个像素对应地面的实际距离 GSD = (H*S)/F
		if img_A.Pixsize is None or img_A.FocalLength is None or img_A.height is None:
			print("____________________使用了EstimatedGSD_____________________")
			GSD = EstimatedGSD
		else:
			GSD = cv_util.get_GSD(img_A.Pixsize, img_A.FocalLength, img_A.height, self.scale_images)#像素大小（像元微米）、相机焦距（毫米）、飞行高度（米）


		x = (cv_util.get_gps_distance(img_A.lat,img_A.lon,img_A.lat,img_B.lon)+cv_util.get_gps_distance(img_B.lat,img_A.lon,img_B.lat,img_B.lon))/2
		print("x实际上移动的距离为{0}".format(x))
		if img_A.lon > img_B.lon:
			x = x /GSD
		else:
			x = -x /GSD
		
		y = cv_util.get_gps_distance(img_A.lat,img_A.lon,img_B.lat,img_A.lon)
		print("y实际上移动的距离为{0}".format(y))
		if img_A.lat > img_B.lat:
			y = -y /GSD
		else:
			y = y /GSD
		
		return x,y,theta	
	def MegaStitchSimilarityAffine(self, similarity):

		V = np.eye(6*self.num_images)

		A = []
		b = []

		tr_c = self.coefs[0]
		gps_c = self.coefs[1]
		co_c = self.coefs[2]

		off_transform_penalty = 6*self.num_images
		print(tr_c,gps_c,co_c,self.max_matches_to_use)
## FTP-Stitch Method
		EstimatedGSD = self.Estimate_GSD()
		X_run = 0
		Y_run = 0
		for image in self.images:

			img_A_name = image.name
			img_B_name = self.image_reference_name
			# print(img_A_name, img_A_name)
			if img_A_name == img_B_name:
				continue
			if self.images_dict[img_A_name].lat == None:
				continue
			T_A_11 = 6*self.n_to_i_dict[img_A_name]
			T_A_12 = 6*self.n_to_i_dict[img_A_name]+1
			T_A_13 = 6*self.n_to_i_dict[img_A_name]+2
			T_A_21 = 6*self.n_to_i_dict[img_A_name]+3
			T_A_22 = 6*self.n_to_i_dict[img_A_name]+4
			T_A_23 = 6*self.n_to_i_dict[img_A_name]+5

			T_B_11 = 6*self.n_to_i_dict[img_B_name]
			T_B_12 = 6*self.n_to_i_dict[img_B_name]+1
			T_B_13 = 6*self.n_to_i_dict[img_B_name]+2
			T_B_21 = 6*self.n_to_i_dict[img_B_name]+3
			T_B_22 = 6*self.n_to_i_dict[img_B_name]+4
			T_B_23 = 6*self.n_to_i_dict[img_B_name]+5
			if (self.no_gps_distance == False):
				X_run,Y_run,Theta = self.Get_Pix_DistanceAndAngle(img_A_name,img_B_name,EstimatedGSD)
				print("————————————————在X方向移动的像素距离为{0}, 在Y方向移动的距离为{1}".format(X_run, Y_run))
			# x_run = 1668
			# Y_run = 32
			else:
				# 直接使用实际距离（相似三角形约束）每两张图片都是一样的X_run和Y_run
				X_run = self.X_run
				Y_run = self.Y_run

			if gps_c != 0:
				width1 = self.width/2
				height1 = self.height/2
				row = gps_c*((V[T_A_11]*(width1) + V[T_A_12]*(height1) + V[T_A_13]*1) - (V[T_B_11]*(width1) + V[T_B_12]*(height1) + V[T_B_13]*1))
				A.append(row)
				b.append(gps_c*X_run)
				row = gps_c * ((V[T_A_21]*(width1) + V[T_A_22]*(height1) + V[T_A_23]*1) - (V[T_B_21]*(width1) + V[T_B_22]*(height1) + V[T_B_23]*1))
				A.append(row)
				b.append(gps_c*Y_run)

		select_matches_all = {}
		for img_A_name in self.pairwise_transformations:
			
			T_A_11 = 6*self.n_to_i_dict[img_A_name]
			T_A_12 = 6*self.n_to_i_dict[img_A_name]+1
			T_A_13 = 6*self.n_to_i_dict[img_A_name]+2
			T_A_21 = 6*self.n_to_i_dict[img_A_name]+3
			T_A_22 = 6*self.n_to_i_dict[img_A_name]+4
			T_A_23 = 6*self.n_to_i_dict[img_A_name]+5

			for img_B_name in self.pairwise_transformations[img_A_name]:
				# print(img_A_name, img_B_name)
				T_B_11 = 6*self.n_to_i_dict[img_B_name]
				T_B_12 = 6*self.n_to_i_dict[img_B_name]+1
				T_B_13 = 6*self.n_to_i_dict[img_B_name]+2
				T_B_21 = 6*self.n_to_i_dict[img_B_name]+3
				T_B_22 = 6*self.n_to_i_dict[img_B_name]+4
				T_B_23 = 6*self.n_to_i_dict[img_B_name]+5

				matches = self.pairwise_transformations[img_A_name][img_B_name][1]
				inliers = self.pairwise_transformations[img_A_name][img_B_name][3]
				bins = self.pairwise_transformations[img_A_name][img_B_name][4]
				numm = len(inliers)
				numi = np.sum(inliers)
				perc = round(numi/numm,2)
				# print(img_A_name, img_B_name, self.n_to_i_dict[img_A_name], self.n_to_i_dict[img_B_name])
				# if gps_c != 0:
					# row = gps_c*(V[T_A_11]*(self.width/2) + V[T_A_12]*(self.height/2) + V[T_A_13]*1 - (V[T_B_11]*(self.width/2) + V[T_B_12]*(self.height/2) + V[T_B_13]*1))
					# A.append(row)
					# b.append(gps_c*X_run)
					# row = gps_c * (V[T_A_21]*(self.width/2) + V[T_A_22]*(self.height/2) + V[T_A_23]*1 - (V[T_B_21]*(self.width/2) + V[T_B_22]*(self.height/2) + V[T_B_23]*1))
					# A.append(row)
					# b.append(gps_c*Y_run)
				if self.cross_validation_k == -1 or bins is None:
					
					inlier_counter = 0
					# print("在X方向移动的像素距离为{0}, 在Y方向移动的距离为{1}".format(X_run, Y_run))
					# print("可最大使用内点个数为{0}".format(self.max_matches_to_use))
					# print("matches匹配对数组的长度为{0}".format(len(matches)))

					# 根据

					# 根据网格平均匹配对
					kp_2 = []
					kp_1 = []
					if self.keypoint == 4:
						kp_1 = self.images_dict[img_A_name].kp[self.n_to_i_dict[img_B_name]]
						kp_2 = self.images_dict[img_B_name].kp[self.n_to_i_dict[img_A_name]]
					else:
						kp_1 = self.images_dict[img_A_name].kp
						kp_2 = self.images_dict[img_B_name].kp
					H = self.pairwise_transformations[img_A_name][img_B_name][0]
					img2_shape = [self.height, self.width]
					# 1600 -1700
					X_run,Y_run,Theta = self.Get_Pix_DistanceAndAngle(img_A_name,img_B_name,EstimatedGSD)
					selected_matches = cv_util.grid_based_point_selection(matches, inliers, H, img2_shape, kp_1, kp_2, self.grid_size, self.max_matches_to_use, self.scale_images, self.n_to_i_dict[img_A_name], self.n_to_i_dict[img_B_name], X_run, Y_run)
					# print("selected_matches = {0}".format(len(selected_matches)))

					# 存储selected_matches
					if img_A_name not in select_matches_all:
						select_matches_all[img_A_name] = {}
					if img_B_name not in select_matches_all:
						select_matches_all[img_B_name] = {}
					select_matches_all[img_A_name][img_B_name] = selected_matches
					
					for i,m in enumerate(selected_matches):

						# if inliers[i,0] == 0:
						# 	# print("inliers[i,0] == 0")
						# 	continue
						
						kp_A = []
						kp_B = []
						if self.keypoint == 4:
							# print(img_A_name, self.n_to_i_dict[img_A_name])
							# print(img_B_name, self.n_to_i_dict[img_B_name])
							kp_A = self.images_dict[img_A_name].kp[self.n_to_i_dict[img_B_name]][m.trainIdx]
							kp_B = self.images_dict[img_B_name].kp[self.n_to_i_dict[img_A_name]][m.queryIdx]
							# if (abs(self.n_to_i_dict[img_A_name]-self.n_to_i_dict[img_B_name]) == 1) and(abs(kp_A[0]-kp_B[0]) > 1700 or abs(kp_A[0]-kp_B[0]) < 1600):
							# 	continue
							# if (abs(self.n_to_i_dict[img_A_name] - self.n_to_i_dict[img_B_name]) > 1) and (abs(kp_A[1]-kp_B[1]) > 1700 or abs(kp_A[1]-kp_B[1]) < 1600):
							# 	continue
						else:
							kp_A = self.images_dict[img_A_name].kp[m.trainIdx]
							kp_B = self.images_dict[img_B_name].kp[m.queryIdx]

						# 筛选内点
						

						p_A = (kp_A[0],kp_A[1])
						p_B = (kp_B[0],kp_B[1])

						# dx = abs(p_B[0]-p_A[0])
						# dy = abs(p_B[1]-p_A[1])
						# # print(dx, dy)
						# if (dx > X_run + 30 or dy > Y_run + 30):
						# 	continue

						row = tr_c*(V[T_A_11]*p_A[0] + V[T_A_12]*p_A[1] + V[T_A_13]*1 - (V[T_B_11]*p_B[0] + V[T_B_12]*p_B[1] + V[T_B_13]*1))
						A.append(row)
						b.append(0)

						row = tr_c*(V[T_A_21]*p_A[0] + V[T_A_22]*p_A[1] + V[T_A_23]*1 - (V[T_B_21]*p_B[0] + V[T_B_22]*p_B[1] + V[T_B_23]*1))
						A.append(row)
						b.append(0)

						inlier_counter+=1

						if inlier_counter >= self.max_matches_to_use:
							break
					print("使用的内点个数为{0}".format(inlier_counter))
				else:

					for m in bins[self.cross_validation_k]:

						kp_A = []
						kp_B = []
						if self.keypoint == 4:
							kp_A = self.images_dict[img_A_name].kp[self.n_to_i_dict[image_B_name]][m.trainIdx]
							kp_B = self.images_dict[img_B_name].kp[self.n_to_i_dict[image_A_name]][m.queryIdx]
						else:
							kp_A = self.images_dict[img_A_name].kp[m.trainIdx]
							kp_B = self.images_dict[img_B_name].kp[m.queryIdx]

						p_A = (kp_A[0],kp_A[1])
						p_B = (kp_B[0],kp_B[1])

						row = tr_c*(V[T_A_11]*p_A[0] + V[T_A_12]*p_A[1] + V[T_A_13]*1 - (V[T_B_11]*p_B[0] + V[T_B_12]*p_B[1] + V[T_B_13]*1))
						A.append(row)
						b.append(0)

						row = tr_c*(V[T_A_21]*p_A[0] + V[T_A_22]*p_A[1] + V[T_A_23]*1 - (V[T_B_21]*p_B[0] + V[T_B_22]*p_B[1] + V[T_B_23]*1))
						A.append(row)
						b.append(0)



		for img_name in self.n_to_i_dict:

			T_A_11 = 6*self.n_to_i_dict[img_name]
			T_A_12 = 6*self.n_to_i_dict[img_name]+1
			T_A_13 = 6*self.n_to_i_dict[img_name]+2
			T_A_21 = 6*self.n_to_i_dict[img_name]+3
			T_A_22 = 6*self.n_to_i_dict[img_name]+4
			T_A_23 = 6*self.n_to_i_dict[img_name]+5


			if img_name == self.image_reference_name:
				if gps_c ==0:
					pass
				A.append(off_transform_penalty*(V[T_A_11]))
				b.append(off_transform_penalty*1)

				A.append(off_transform_penalty*(V[T_A_12]))
				b.append(off_transform_penalty*0)
	
				A.append(off_transform_penalty*(V[T_A_21]))
				b.append(off_transform_penalty*0)

				A.append(off_transform_penalty*(V[T_A_22]))
				b.append(off_transform_penalty*1)

				A.append(off_transform_penalty*(V[T_A_13]))
				b.append(off_transform_penalty*0)

				A.append(off_transform_penalty*(V[T_A_23]))
				b.append(off_transform_penalty*0)
    
				A.append(off_transform_penalty*(V[T_A_11] - V[T_A_22]))
				b.append(0)

				A.append(off_transform_penalty*(V[T_A_12] + V[T_A_21]))
				b.append(0)
				# A.append(off_transform_penalty*(V[T_A_21]))
				# b.append(off_transform_penalty*0)

				# A.append(off_transform_penalty*(V[T_A_12]))
				# b.append(off_transform_penalty*0)


			else:
				if similarity:

					A.append(off_transform_penalty*(V[T_A_11] - V[T_A_22]))
					b.append(0)

					A.append(off_transform_penalty*(V[T_A_12] + V[T_A_21]))
					b.append(0)
					# A.append(off_transform_penalty*(V[T_A_21]))
					# b.append(off_transform_penalty*0)

					# A.append(off_transform_penalty*(V[T_A_12]))
					# b.append(off_transform_penalty*0)


		A = np.array(A)
		b = np.array(b)

		m = 1e-4
		lower_bounds = [-np.inf]*(6*self.num_images)
		upper_bounds = [np.inf]*(6*self.num_images)
		# lower_bounds = [-1]*(2*self.num_images) + [-np.inf]*(1*self.num_images) + [-1]*(2*self.num_images) + [-np.inf]*(1*self.num_images)
		# upper_bounds = [1]*(2*self.num_images) + [np.inf]*(1*self.num_images) + [1]*(2*self.num_images) + [np.inf]*(1*self.num_images)
		
		i = self.n_to_i_dict[self.image_reference_name]

		# Sin = math.sin(math.pi/180*self.images_dict[self.image_reference_name].Yaw) 
		# Cos = math.cos(math.pi/180*self.images_dict[self.image_reference_name].Yaw)
		# Sin = 0
		# Cos = 1	

		# lower_bounds[6*i+0] = Cos-m
		# upper_bounds[6*i+0] = Cos+m

		# lower_bounds[6*i+1] = -Sin-m
		# upper_bounds[6*i+1] = -Sin+m



		# lower_bounds[6*i+3] = Sin-m
		# upper_bounds[6*i+3] = Sin+m

		# lower_bounds[6*i+4] = Cos-m
		# upper_bounds[6*i+4] = Cos+m
		# lower_bounds[6*i+0] = Cos-m
		# upper_bounds[6*i+0] = Cos+m
  
		lower_bounds[6*i+2] = 0-m
		upper_bounds[6*i+2] = 0+m
  
		lower_bounds[6*i+5] = 0-m
		upper_bounds[6*i+5] = 0+m
		
		start_time = datetime.datetime.now()
		
		# res = lsq_linear(A, b, max_iter=4*self.num_images,verbose=0)
		res = lsq_linear(A, b, bounds=(lower_bounds,upper_bounds), max_iter=4*self.num_images,verbose=2)
		
		end_time = datetime.datetime.now()
		report_time(start_time,end_time)

		X = res.x

		residuals = res.fun
		
		print(':: Least Squares RMSE: {0}'.format(np.sqrt(np.mean(residuals**2))))
		## i am calculating GPSRMSE ing ..............
		GPS_residuals = []
		for image in self.images:
			residual = 0
			img_A_name = image.name
			img_B_name = self.image_reference_name
			if img_A_name == img_B_name:
				GPS_residuals.append(residual)
			if self.images_dict[img_A_name].lat == None:
				continue
			T_A_11 = 6*self.n_to_i_dict[img_A_name]
			T_A_12 = 6*self.n_to_i_dict[img_A_name]+1
			T_A_13 = 6*self.n_to_i_dict[img_A_name]+2
			T_A_21 = 6*self.n_to_i_dict[img_A_name]+3
			T_A_22 = 6*self.n_to_i_dict[img_A_name]+4
			T_A_23 = 6*self.n_to_i_dict[img_A_name]+5

			T_B_11 = 6*self.n_to_i_dict[img_B_name]
			T_B_12 = 6*self.n_to_i_dict[img_B_name]+1
			T_B_13 = 6*self.n_to_i_dict[img_B_name]+2
			T_B_21 = 6*self.n_to_i_dict[img_B_name]+3
			T_B_22 = 6*self.n_to_i_dict[img_B_name]+4
			T_B_23 = 6*self.n_to_i_dict[img_B_name]+5
			X_run,Y_run,Theta = self.Get_Pix_DistanceAndAngle(img_A_name,img_B_name,EstimatedGSD)
			# X_run = 1668
			# Y_run = 32

			# d_x = ( X[T_A_13]*1 - X[T_B_13]*1 - X_run)
			# GPS_residuals.append(d_x**2)

			# d_y = ( X[T_A_23]*1 - X[T_B_23]*1 - Y_run)
			# GPS_residuals.append(d_y**2)

			d_x = ( X[T_A_13]*1 - X[T_B_13]*1)

			d_y = ( X[T_A_23]*1 - X[T_B_23]*1)

			# print(d_x,d_y,X_run,Y_run)

			d2 = math.sqrt(d_x ** 2 + d_y ** 2)
			
			X2 = math.sqrt(X_run ** 2 + Y_run ** 2)

			GPS_residuals.append(abs(X2-d2))
		GPSRMSE = math.sqrt(np.mean(GPS_residuals))

		# 	d_x = ( X[T_A_13]*1 - X[T_B_13]*1 - X_run)
		# 	GPS_residuals.append(d_x**2)
		# 	d_y = ( X[T_A_23]*1 - X[T_B_23]*1 - Y_run)
		# 	GPS_residuals.append(d_y**2)
		# 	# print(GPS_residuals)
		# GPSRMSE = math.sqrt(np.mean(GPS_residuals))
		print(GPSRMSE)
		print(':: Least Squares GPSRMSE: {0}'.format(GPSRMSE))
		## end calculating GPSRMSE ing ..............

		# ## i am calculating MatchingRMSE ing ..............
		# Matching_residuals = []
		# for img_A_name in self.pairwise_transformations:

		# 	T_A_11 = 6*self.n_to_i_dict[img_A_name]
		# 	T_A_12 = 6*self.n_to_i_dict[img_A_name]+1
		# 	T_A_13 = 6*self.n_to_i_dict[img_A_name]+2
		# 	T_A_21 = 6*self.n_to_i_dict[img_A_name]+3
		# 	T_A_22 = 6*self.n_to_i_dict[img_A_name]+4
		# 	T_A_23 = 6*self.n_to_i_dict[img_A_name]+5
		# 	T_A = np.eye(3)
		# 	T_A[0,0] = X[T_A_11]
		# 	T_A[0,1] = X[T_A_12]
		# 	T_A[0,2] = X[T_A_13]
		# 	T_A[1,0] = X[T_A_21]
		# 	T_A[1,1] = X[T_A_22]
		# 	T_A[1,2] = X[T_A_23]

		# 	for img_B_name in self.pairwise_transformations[img_A_name]:
		
		# 		T_B_11 = 6*self.n_to_i_dict[img_B_name]
		# 		T_B_12 = 6*self.n_to_i_dict[img_B_name]+1
		# 		T_B_13 = 6*self.n_to_i_dict[img_B_name]+2
		# 		T_B_21 = 6*self.n_to_i_dict[img_B_name]+3
		# 		T_B_22 = 6*self.n_to_i_dict[img_B_name]+4
		# 		T_B_23 = 6*self.n_to_i_dict[img_B_name]+5
		# 		T_B = np.eye(3)
		# 		T_B[0,0] = X[T_B_11]
		# 		T_B[0,1] = X[T_B_12]
		# 		T_B[0,2] = X[T_B_13]
		# 		T_B[1,0] = X[T_B_21]
		# 		T_B[1,1] = X[T_B_22]
		# 		T_B[1,2] = X[T_B_23]

		# 		matches = self.pairwise_transformations[img_A_name][img_B_name][1]
		# 		inliers = self.pairwise_transformations[img_A_name][img_B_name][3]
		# 		bins = self.pairwise_transformations[img_A_name][img_B_name][4]
		# 		numm = len(inliers)
		# 		numi = np.sum(inliers)
		# 		perc = round(numi/numm,2)

		# 		for i, m in enumerate(matches):

		# 			if inliers[i, 0] == 0:
		# 				continue

		# 			kp_A = self.images_dict[img_A_name].kp[m.trainIdx]
		# 			kp_B = self.images_dict[img_B_name].kp[m.queryIdx]

		# 			p_A = [kp_A[0], kp_A[1], 1]
		# 			p_B = [kp_B[0], kp_B[1], 1]


		# 			new_p_A = np.matmul(T_A, p_A)
		# 			if T_A.shape[0] == 3:
		# 				new_p_A = new_p_A / new_p_A[2]
		# 			new_p_B = np.matmul(T_B, p_B)
		# 			if T_B.shape[0] == 3:
		# 				new_p_B = new_p_B / new_p_B[2]

		# 			Matching_residuals.append((new_p_A[0] - new_p_B[0]) ** 2)
		# 			Matching_residuals.append((new_p_A[1] - new_p_B[1]) ** 2)

		# MatchingRMSE = math.sqrt(np.mean(Matching_residuals))
		# print(MatchingRMSE)
		# print(':: Least Squares MatchingRMSE: {0}'.format(MatchingRMSE))
		# ## end calculating MatchingRMSE ing ..............

		image_corners_dict = {}
		absolute_transformations_dict = {}

		T_ref_11 = 6*self.n_to_i_dict[self.image_reference_name]
		T_ref_12 = 6*self.n_to_i_dict[self.image_reference_name]+1
		T_ref_13 = 6*self.n_to_i_dict[self.image_reference_name]+2
		T_ref_21 = 6*self.n_to_i_dict[self.image_reference_name]+3
		T_ref_22 = 6*self.n_to_i_dict[self.image_reference_name]+4
		T_ref_23 = 6*self.n_to_i_dict[self.image_reference_name]+5
		H_ref = np.eye(3)
		H_ref[0,0] = X[T_ref_11]
		H_ref[0,1] = X[T_ref_12]
		H_ref[0,2] = X[T_ref_13]
		H_ref[1,0] = X[T_ref_21]
		H_ref[1,1] = X[T_ref_22]
		H_ref[1,2] = X[T_ref_23]

		for img_name in self.n_to_i_dict:
			img_A_name = img_name
			img_B_name = self.image_reference_name
			X_run,Y_run,Theta = self.Get_Pix_DistanceAndAngle(img_A_name,img_B_name,EstimatedGSD)
			T_A_11 = 6*self.n_to_i_dict[img_name]
			T_A_12 = 6*self.n_to_i_dict[img_name]+1
			T_A_13 = 6*self.n_to_i_dict[img_name]+2
			T_A_21 = 6*self.n_to_i_dict[img_name]+3
			T_A_22 = 6*self.n_to_i_dict[img_name]+4
			T_A_23 = 6*self.n_to_i_dict[img_name]+5

			T = np.eye(3)
			T[0,0] = X[T_A_11]
			T[0,1] = X[T_A_12]
			T[0,2] = X[T_A_13]
			T[1,0] = X[T_A_21]
			T[1,1] = X[T_A_22]
			T[1,2] = X[T_A_23]

			UL_ref = [0,0,1]
			UR_ref = [self.width,0,1]
			LR_ref = [self.width,self.height,1]
			LL_ref = [0,self.height,1]
			center_ref = [self.width/2,self.height/2,1]

			UL = np.matmul(T,UL_ref)
			UL = UL/UL[2]
			UL = UL[:2]

			UR = np.matmul(T,UR_ref)
			UR = UR/UR[2]
			UR = UR[:2]

			LR = np.matmul(T,LR_ref)
			LR = LR/LR[2]
			LR = LR[:2]

			LL = np.matmul(T,LL_ref)
			LL = LL/LL[2]
			LL = LL[:2]

			image_corners_dict[img_name] = {'UL':UL,'UR':UR,'LR':LR,'LL':LL}
			CenterX = (UL[0]+UR[0]+LL[0]+LR[0])/4
			CenterY = (UL[1]+UR[1]+LL[1]+LR[1])/4
			# print(CenterX,CenterY)
			# print(np.matmul(T,center_ref)-np.matmul(H_ref,center_ref))

			# print(X_run,Y_run)

			absolute_transformations_dict[img_name] = T
			# print(f"{img_name}:{math.degrees(math.atan(T[1,0]/T[0,0]))}--{self.images_dict[img_name].Yaw}--{T[1,0]**2+T[0,0]**2}")

		return image_corners_dict,absolute_transformations_dict, end_time-start_time, select_matches_all

	def phi(self,k,x,y):

		if k == 0:
			# UL
			return (1-(y/self.height))*(1-(x/self.width))
		elif k == 1:
			# UR
			return (1-(y/self.height))*(x/self.width)
		elif k == 2:
			# LL
			return (y/self.height)*(1-(x/self.width))
		elif k == 3:
			# LR
			return (y/self.height)*(x/self.width)

	def MegaStitchTranslationKeyPointBased(self,anchors,noisy_corners,no_anchor_use=1):

		C = np.eye(4*self.num_images)

		A = []
		b = []

		GPS_c = self.coefs[1]

		off = GPS_c*self.num_images
		
		for img_A_name in self.pairwise_transformations:

			Corner_A = {\
			'UL':[4*self.n_to_i_dict[img_A_name]+0,4*self.n_to_i_dict[img_A_name]+1],\
			'UR':[4*self.n_to_i_dict[img_A_name]+2,4*self.n_to_i_dict[img_A_name]+1],\
			'LL':[4*self.n_to_i_dict[img_A_name]+0,4*self.n_to_i_dict[img_A_name]+3],\
			'LR':[4*self.n_to_i_dict[img_A_name]+2,4*self.n_to_i_dict[img_A_name]+3]}

			for img_B_name in self.pairwise_transformations[img_A_name]:
				
				Corner_B = {\
				'UL':[4*self.n_to_i_dict[img_B_name]+0,4*self.n_to_i_dict[img_B_name]+1],\
				'UR':[4*self.n_to_i_dict[img_B_name]+2,4*self.n_to_i_dict[img_B_name]+1],\
				'LL':[4*self.n_to_i_dict[img_B_name]+0,4*self.n_to_i_dict[img_B_name]+3],\
				'LR':[4*self.n_to_i_dict[img_B_name]+2,4*self.n_to_i_dict[img_B_name]+3]}

				matches = self.pairwise_transformations[img_A_name][img_B_name][1]
				inliers = self.pairwise_transformations[img_A_name][img_B_name][3]

				num_inliers = min(np.sum(inliers),self.max_matches_to_use)

				EQ_c = self.coefs[0]
				
				inlier_counter = 0

				for i,m in enumerate(matches):

					if inliers[i,0] == 0:
						continue

					kp_A = []
					kp_B = []
					if self.keypoint == 4:
						kp_A = self.images_dict[img_A_name].kp[self.n_to_i_dict[image_B_name]][m.trainIdx]
						kp_B = self.images_dict[img_B_name].kp[self.n_to_i_dict[image_A_name]][m.queryIdx]
					else:
						kp_A = self.images_dict[img_A_name].kp[m.trainIdx]
						kp_B = self.images_dict[img_B_name].kp[m.queryIdx]

					p_A = (kp_A[0],kp_A[1])
					p_B = (kp_B[0],kp_B[1])

					row_x = np.zeros(4*self.num_images)
					row_y = np.zeros(4*self.num_images)

					for i,k in enumerate(['UL','UR','LL','LR']):

						row_x += (self.phi(i,p_A[0],p_A[1])*C[Corner_A[k][0]] - self.phi(i,p_B[0],p_B[1])*C[Corner_B[k][0]])
						row_y += (self.phi(i,p_A[0],p_A[1])*C[Corner_A[k][1]] - self.phi(i,p_B[0],p_B[1])*C[Corner_B[k][1]])

					A.append(EQ_c*row_x)
					A.append(EQ_c*row_y)
					b.append(0)
					b.append(0)
					
					inlier_counter+=1

					if inlier_counter >= self.max_matches_to_use:
						break

		used_anchors = 0
		
		for img_name in self.n_to_i_dict:

			Corner = {\
			'UL':[4*self.n_to_i_dict[img_name]+0,4*self.n_to_i_dict[img_name]+1],\
			'UR':[4*self.n_to_i_dict[img_name]+2,4*self.n_to_i_dict[img_name]+1],\
			'LL':[4*self.n_to_i_dict[img_name]+0,4*self.n_to_i_dict[img_name]+3],\
			'LR':[4*self.n_to_i_dict[img_name]+2,4*self.n_to_i_dict[img_name]+3]}

			# for i,k in enumerate(['UL','UR','LL','LR']):
			for i,k in enumerate(['UL','LR']):

				A.append(GPS_c*C[Corner[k][0]])
				b.append(GPS_c*noisy_corners[img_name][k]['lon'])

				A.append(GPS_c*C[Corner[k][1]])
				b.append(-1*GPS_c*noisy_corners[img_name][k]['lat'])

			A.append(off*(C[Corner['UR'][0]]-C[Corner['UL'][0]]))
			b.append(off*(noisy_corners[img_name]['UR']['lon']-noisy_corners[img_name]['UL']['lon']))

			A.append(off*(C[Corner['LL'][1]]-C[Corner['UL'][1]]))
			b.append(off*(noisy_corners[img_name]['UL']['lat']-noisy_corners[img_name]['LL']['lat']))

			if img_name in anchors and used_anchors<no_anchor_use:

				A_w = anchors[img_name]['img_x']*self.scale_images
				A_h = anchors[img_name]['img_y']*self.scale_images
				A_lon = anchors[img_name]['gps_lon']
				A_lat = -1*anchors[img_name]['gps_lat']

				row_x = np.zeros(4*self.num_images)
				row_y = np.zeros(4*self.num_images)

				for i,k in enumerate(['UL','UR','LL','LR']):

					row_x += (self.phi(i,A_w,A_h)*C[Corner[k][0]])
					row_y += (self.phi(i,A_w,A_h)*C[Corner[k][1]])

				A.append(off*row_x)
				b.append(off*A_lon)

				A.append(off*row_y)
				b.append(off*A_lat)

				used_anchors += 1

		start_time = datetime.datetime.now()
		
		res = lsq_linear(A, b, max_iter=4*self.num_images,verbose=0)
		
		end_time = datetime.datetime.now()
		report_time(start_time,end_time)

		X = res.x

		residuals = res.fun
		
		print('\tRMSE: {0}'.format(np.sqrt(np.mean(residuals**2))))

		image_corners_dict = {}

		for img_name in self.n_to_i_dict:

			Corner = {\
			'UL':[4*self.n_to_i_dict[img_name]+0,4*self.n_to_i_dict[img_name]+1],\
			'UR':[4*self.n_to_i_dict[img_name]+2,4*self.n_to_i_dict[img_name]+1],\
			'LL':[4*self.n_to_i_dict[img_name]+0,4*self.n_to_i_dict[img_name]+3],\
			'LR':[4*self.n_to_i_dict[img_name]+2,4*self.n_to_i_dict[img_name]+3]}

			UL = [X[Corner['UL'][0]],X[Corner['UL'][1]]]

			UR = [X[Corner['UR'][0]],X[Corner['UR'][1]]]

			LR = [X[Corner['LR'][0]],X[Corner['LR'][1]]]

			LL = [X[Corner['LL'][0]],X[Corner['LL'][1]]]

			image_corners_dict[img_name] = {'UL':UL,'UR':UR,'LR':LR,'LL':LL}

		return image_corners_dict

	def MegaStitchTranslationParameterBased(self,anchors,noisy_corners,no_anchor_use=1):
		
		GPS_Width = noisy_corners[self.image_reference_name]['UR']['lon']-noisy_corners[self.image_reference_name]['UL']['lon']
		GPS_Height = noisy_corners[self.image_reference_name]['UL']['lat']-noisy_corners[self.image_reference_name]['LL']['lat']
		GPS_IMG_Ratio = (GPS_Width/self.width,GPS_Height/self.height)

		C = np.eye(2*self.num_images)

		A = []
		b = []

		GPS_c = self.coefs[1]

		off = GPS_c*self.num_images
		
		for img_A_name in self.pairwise_transformations:

			for img_B_name in self.pairwise_transformations[img_A_name]:
				
				translation = self.pairwise_transformations[img_A_name][img_B_name][0]
				inliers = self.pairwise_transformations[img_A_name][img_B_name][3]

				perc = np.sum(inliers)/len(inliers)

				if perc<0.4:
					continue

				translation = get_translation_in_GPS_coordinate_system(translation,noisy_corners[self.image_reference_name],self.width,self.height)

				EQ_c = self.coefs[0]

				row_x = - EQ_c*C[2*self.n_to_i_dict[img_A_name],:] + EQ_c*C[2*self.n_to_i_dict[img_B_name],:]
				row_y = - EQ_c*C[2*self.n_to_i_dict[img_A_name]+1,:] + EQ_c*C[2*self.n_to_i_dict[img_B_name]+1,:]

				A.append(EQ_c*row_x)
				A.append(EQ_c*row_y)
				b.append(EQ_c*translation[0])
				b.append(EQ_c*translation[1])

		used_anchors = 0
		
		for img_name in self.n_to_i_dict:

			Corner = (2*self.n_to_i_dict[img_name],2*self.n_to_i_dict[img_name]+1)

			A.append(GPS_c*C[Corner[0]])
			b.append(GPS_c*noisy_corners[img_name]['UL']['lon'])

			A.append(GPS_c*C[Corner[1]])
			b.append(GPS_c*noisy_corners[img_name]['UL']['lat'])
			
			if img_name in anchors and used_anchors<no_anchor_use:

				A_w = anchors[img_name]['img_x']*self.scale_images
				A_h = anchors[img_name]['img_y']*self.scale_images
				A_lon = anchors[img_name]['gps_lon']+A_w*GPS_IMG_Ratio[0]
				A_lat = anchors[img_name]['gps_lat']-A_h*GPS_IMG_Ratio[1]

				row_x = C[Corner[0]]
				row_y = C[Corner[1]]

				A.append(off*row_x)
				b.append(off*A_lon)

				A.append(off*row_y)
				b.append(off*A_lat)

				used_anchors += 1


		start_time = datetime.datetime.now()
	  
		res = lsq_linear(A, b, max_iter=2*self.num_images,verbose=0)

		end_time = datetime.datetime.now()
		report_time(start_time,end_time)
		
		X = res.x
		
		residuals = res.fun
		
		print('\tRMSE: {0}'.format(np.sqrt(np.mean(residuals**2))))

		image_corners_dict = {}

		for img_name in self.n_to_i_dict:

			ULCorner =[2*self.n_to_i_dict[img_name],2*self.n_to_i_dict[img_name]+1]

			UL = [X[ULCorner[0]],X[ULCorner[1]]]

			UR = [X[ULCorner[0]]+GPS_Width,X[ULCorner[1]]]

			LR = [X[ULCorner[0]]+GPS_Width,X[ULCorner[1]]-GPS_Height]

			LL = [X[ULCorner[0]],X[ULCorner[1]]-GPS_Height]

			image_corners_dict[img_name] = {'UL':UL,'UR':UR,'LR':LR,'LL':LL}

		return image_corners_dict

	def get_residuals(self,X):

		residuals = []
		off_transform_penalty = sys.maxsize

		for img1_name in self.pairwise_transformations:

			for img2_name in self.pairwise_transformations[img1_name]:
				
				matches = self.pairwise_transformations[img1_name][img2_name][1]
				inliers = self.pairwise_transformations[img1_name][img2_name][3]
				bins = self.pairwise_transformations[img1_name][img2_name][4]

				i = self.n_to_i_dict[img1_name]
				j = self.n_to_i_dict[img2_name]

				H1 = X[i*9:i*9+9]
				H1 = H1.reshape(3,3)
				
				H2 = X[j*9:j*9+9]
				H2 = H2.reshape(3,3)				
				
				if self.cross_validation_k == -1 or bins is None:
					
					inliers_iterator = 0

					for i,m in enumerate(matches):

						if inliers[i,0] == 0:
							continue

						kp1 = self.images_dict[img1_name].kp[m.trainIdx]
						kp2 = self.images_dict[img2_name].kp[m.queryIdx]

						kp1 = self.images_dict[img1_name].kp[n_to_i_dict[image2_name]][m.trainIdx]
						kp2 = self.images_dict[img2_name].kp[n_to_i_dict[image1_name]][m.queryIdx]

						p1 = [kp1[0],kp1[1],1]
						p2 = [kp2[0],kp2[1],1]

						p1_r = np.matmul(H1,p1)
						p1_r = p1_r/p1_r[2]

						p2_r = np.matmul(H2,p2)
						p2_r = p2_r/p2_r[2]

						residuals.append((p1_r[0]-p2_r[0])**2+(p1_r[1]-p2_r[1])**2)

						inliers_iterator+=1

						if inliers_iterator>=self.max_matches_to_use:
							break
				else:

					for m in bins[self.cross_validation_k]:

						kp1 = self.images_dict[img1_name].kp[m.trainIdx]
						kp2 = self.images_dict[img2_name].kp[m.queryIdx]

						kp1 = self.images_dict[img1_name].kp[n_to_i_dict[image2_name]][m.trainIdx]
						kp2 = self.images_dict[img2_name].kp[n_to_i_dict[image1_name]][m.queryIdx]

						p1 = [kp1[0],kp1[1],1]
						p2 = [kp2[0],kp2[1],1]

						p1_r = np.matmul(H1,p1)
						p1_r = p1_r/p1_r[2]

						p2_r = np.matmul(H2,p2)
						p2_r = p2_r/p2_r[2]

						residuals.append((p1_r[0]-p2_r[0])**2+(p1_r[1]-p2_r[1])**2)


		# for img_name in self.images_dict:
		#	 i = self.n_to_i_dict[img_name]

		#	 H1_tmp = X[i*9:i*9+9]
		#	 H1_tmp = H1_tmp.reshape(3,3)

		#	 residuals.append(off_transform_penalty*H1_tmp[2,2]-off_transform_penalty)

		# i = self.n_to_i_dict[self.image_reference_name]
		# H = X[i*9:i*9+9]
		# H = H.reshape(3,3)

		# residuals.append(off_transform_penalty*H[0,1])
		# residuals.append(off_transform_penalty*H[0,2])
		# residuals.append(off_transform_penalty*H[1,0])
		# residuals.append(off_transform_penalty*H[1,2])
		# residuals.append(off_transform_penalty*H[2,0])
		# residuals.append(off_transform_penalty*H[2,1])
		# residuals.append(off_transform_penalty*H[0,0]-off_transform_penalty*1)
		# residuals.append(off_transform_penalty*H[1,1]-off_transform_penalty*1)
		# residuals.append(off_transform_penalty*H[2,2]-off_transform_penalty*1)			

		return residuals

	def get_jacobians(self,X):

		jacobians = []

		off_transform_penalty = sys.maxsize

		for img1_name in self.pairwise_transformations:

			for img2_name in self.pairwise_transformations[img1_name]:
				
				matches = self.pairwise_transformations[img1_name][img2_name][1]
				inliers = self.pairwise_transformations[img1_name][img2_name][3]
				bins = self.pairwise_transformations[img1_name][img2_name][4]

				i = self.n_to_i_dict[img1_name]
				j = self.n_to_i_dict[img2_name]

				H1 = np.eye(3)
				H2 = np.eye(3)

				H1 = X[i*9:i*9+9]
				H1 = H1.reshape(3,3)
				
				H2 = X[j*9:j*9+9]
				H2 = H2.reshape(3,3)				
				
				if self.cross_validation_k == -1 or bins is None:
					
					inliers_iterator = 0

					for i,m in enumerate(matches):

						if inliers[i,0] == 0:
							continue

						kp1 = self.images_dict[img1_name].kp[m.trainIdx]
						kp2 = self.images_dict[img2_name].kp[m.queryIdx]

						kp1 = self.images_dict[img1_name].kp[n_to_i_dict[image2_name]][m.trainIdx]
						kp2 = self.images_dict[img2_name].kp[n_to_i_dict[image1_name]][m.queryIdx]

						p1 = [kp1[0],kp1[1],1]
						p2 = [kp2[0],kp2[1],1]

						p1_r_no_div = np.matmul(H1,p1) 
						p1_r = p1_r_no_div/p1_r_no_div[2]

						p2_r_no_div = np.matmul(H2,p2)
						p2_r = p2_r_no_div/p2_r_no_div[2]

						
						diff_x = p1_r[0]-p2_r[0]
						diff_y = p1_r[1]-p2_r[1]
						residual = (diff_x)**2+(diff_y)**2

						# H11
						rond_x = p1[0]/(p1_r_no_div[2])
						rond_y = 0
						jac_H1_11 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H12
						rond_x = p1[1]/(p1_r_no_div[2])
						rond_y = 0
						jac_H1_12 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H13
						rond_x = 1/(p1_r_no_div[2])
						rond_y = 0
						jac_H1_13 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H21
						rond_x = 0
						rond_y = p1[0]/(p1_r_no_div[2])
						jac_H1_21 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H22
						rond_x = 0
						rond_y = p1[1]/(p1_r_no_div[2])
						jac_H1_22 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H23
						rond_x = 0
						rond_y = 1/(p1_r_no_div[2])
						jac_H1_23 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H31
						rond_x = - p1[0]*p1_r_no_div[0]/((p1_r_no_div[2])**2) 
						rond_y = - p1[0]*p1_r_no_div[1]/((p1_r_no_div[2])**2)
						jac_H1_31 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H32
						rond_x = - p1[1]*p1_r_no_div[0]/((p1_r_no_div[2])**2) 
						rond_y = - p1[1]*p1_r_no_div[1]/((p1_r_no_div[2])**2)
						jac_H1_32 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H33
						rond_x = - 1*p1_r_no_div[0]/((p1_r_no_div[2])**2) 
						rond_y = - 1*p1_r_no_div[1]/((p1_r_no_div[2])**2)
						jac_H1_33 = 2*diff_x*rond_x+2*diff_y*rond_y

						# -----------------

						# H11
						rond_x = -p2[0]/(p2_r_no_div[2])
						rond_y = 0
						jac_H2_11 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H12
						rond_x = -p2[1]/(p2_r_no_div[2])
						rond_y = 0
						jac_H2_12 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H13
						rond_x = -1/(p2_r_no_div[2])
						rond_y = 0
						jac_H2_13 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H21
						rond_x = 0
						rond_y = -p2[0]/(p2_r_no_div[2])
						jac_H2_21 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H22
						rond_x = 0
						rond_y = -p2[1]/(p2_r_no_div[2])
						jac_H2_22 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H23
						rond_x = 0
						rond_y = -1/(p2_r_no_div[2])
						jac_H2_23 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H31
						rond_x = p2[0]*p2_r_no_div[0]/((p2_r_no_div[2])**2) 
						rond_y = p2[0]*p2_r_no_div[1]/((p2_r_no_div[2])**2)
						jac_H2_31 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H32
						rond_x = p2[1]*p2_r_no_div[0]/((p2_r_no_div[2])**2) 
						rond_y = p2[1]*p2_r_no_div[1]/((p2_r_no_div[2])**2)
						jac_H2_32 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H33
						rond_x = 1*p2_r_no_div[0]/((p2_r_no_div[2])**2) 
						rond_y = 1*p2_r_no_div[1]/((p2_r_no_div[2])**2)
						jac_H2_33 = 2*diff_x*rond_x+2*diff_y*rond_y

						ii = self.n_to_i_dict[img1_name]
						jj = self.n_to_i_dict[img2_name]
						jac = np.zeros(9*self.num_images)
						jac[ii*9+0] = jac_H1_11
						jac[ii*9+1] = jac_H1_12
						jac[ii*9+2] = jac_H1_13
						jac[ii*9+3] = jac_H1_21
						jac[ii*9+4] = jac_H1_22
						jac[ii*9+5] = jac_H1_23
						jac[ii*9+6] = jac_H1_31
						jac[ii*9+7] = jac_H1_32
						jac[ii*9+8] = jac_H1_33

						jac[jj*9+0] = jac_H2_11
						jac[jj*9+1] = jac_H2_12
						jac[jj*9+2] = jac_H2_13
						jac[jj*9+3] = jac_H2_21
						jac[jj*9+4] = jac_H2_22
						jac[jj*9+5] = jac_H2_23
						jac[jj*9+6] = jac_H2_31
						jac[jj*9+7] = jac_H2_32
						jac[jj*9+8] = jac_H2_33
						
						jacobians.append(jac)

						inliers_iterator+=1

						if inliers_iterator>=self.max_matches_to_use:
							break
				else:
					
					for m in bins[self.cross_validation_k]:

						kp1 = self.images_dict[img1_name].kp[m.trainIdx]
						kp2 = self.images_dict[img2_name].kp[m.queryIdx]

						kp1 = self.images_dict[img1_name].kp[n_to_i_dict[image2_name]][m.trainIdx]
						kp2 = self.images_dict[img2_name].kp[n_to_i_dict[image1_name]][m.queryIdx]

						p1 = [kp1[0],kp1[1],1]
						p2 = [kp2[0],kp2[1],1]

						p1_r_no_div = np.matmul(H1,p1) 
						p1_r = p1_r_no_div/p1_r_no_div[2]

						p2_r_no_div = np.matmul(H2,p2)
						p2_r = p2_r_no_div/p2_r_no_div[2]

						
						diff_x = p1_r[0]-p2_r[0]
						diff_y = p1_r[1]-p2_r[1]
						residual = (diff_x)**2+(diff_y)**2

						# H11
						rond_x = p1[0]/(p1_r_no_div[2])
						rond_y = 0
						jac_H1_11 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H12
						rond_x = p1[1]/(p1_r_no_div[2])
						rond_y = 0
						jac_H1_12 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H13
						rond_x = 1/(p1_r_no_div[2])
						rond_y = 0
						jac_H1_13 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H21
						rond_x = 0
						rond_y = p1[0]/(p1_r_no_div[2])
						jac_H1_21 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H22
						rond_x = 0
						rond_y = p1[1]/(p1_r_no_div[2])
						jac_H1_22 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H23
						rond_x = 0
						rond_y = 1/(p1_r_no_div[2])
						jac_H1_23 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H31
						rond_x = - p1[0]*p1_r_no_div[0]/((p1_r_no_div[2])**2) 
						rond_y = - p1[0]*p1_r_no_div[1]/((p1_r_no_div[2])**2)
						jac_H1_31 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H32
						rond_x = - p1[1]*p1_r_no_div[0]/((p1_r_no_div[2])**2) 
						rond_y = - p1[1]*p1_r_no_div[1]/((p1_r_no_div[2])**2)
						jac_H1_32 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H33
						rond_x = - 1*p1_r_no_div[0]/((p1_r_no_div[2])**2) 
						rond_y = - 1*p1_r_no_div[1]/((p1_r_no_div[2])**2)
						jac_H1_33 = 2*diff_x*rond_x+2*diff_y*rond_y

						# -----------------

						# H11
						rond_x = -p2[0]/(p2_r_no_div[2])
						rond_y = 0
						jac_H2_11 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H12
						rond_x = -p2[1]/(p2_r_no_div[2])
						rond_y = 0
						jac_H2_12 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H13
						rond_x = -1/(p2_r_no_div[2])
						rond_y = 0
						jac_H2_13 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H21
						rond_x = 0
						rond_y = -p2[0]/(p2_r_no_div[2])
						jac_H2_21 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H22
						rond_x = 0
						rond_y = -p2[1]/(p2_r_no_div[2])
						jac_H2_22 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H23
						rond_x = 0
						rond_y = -1/(p2_r_no_div[2])
						jac_H2_23 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H31
						rond_x = p2[0]*p2_r_no_div[0]/((p2_r_no_div[2])**2) 
						rond_y = p2[0]*p2_r_no_div[1]/((p2_r_no_div[2])**2)
						jac_H2_31 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H32
						rond_x = p2[1]*p2_r_no_div[0]/((p2_r_no_div[2])**2) 
						rond_y = p2[1]*p2_r_no_div[1]/((p2_r_no_div[2])**2)
						jac_H2_32 = 2*diff_x*rond_x+2*diff_y*rond_y

						# H33
						rond_x = 1*p2_r_no_div[0]/((p2_r_no_div[2])**2) 
						rond_y = 1*p2_r_no_div[1]/((p2_r_no_div[2])**2)
						jac_H2_33 = 2*diff_x*rond_x+2*diff_y*rond_y

						ii = self.n_to_i_dict[img1_name]
						jj = self.n_to_i_dict[img2_name]
						jac = np.zeros(9*self.num_images)
						jac[ii*9+0] = jac_H1_11
						jac[ii*9+1] = jac_H1_12
						jac[ii*9+2] = jac_H1_13
						jac[ii*9+3] = jac_H1_21
						jac[ii*9+4] = jac_H1_22
						jac[ii*9+5] = jac_H1_23
						jac[ii*9+6] = jac_H1_31
						jac[ii*9+7] = jac_H1_32
						jac[ii*9+8] = jac_H1_33

						jac[jj*9+0] = jac_H2_11
						jac[jj*9+1] = jac_H2_12
						jac[jj*9+2] = jac_H2_13
						jac[jj*9+3] = jac_H2_21
						jac[jj*9+4] = jac_H2_22
						jac[jj*9+5] = jac_H2_23
						jac[jj*9+6] = jac_H2_31
						jac[jj*9+7] = jac_H2_32
						jac[jj*9+8] = jac_H2_33
						
						jacobians.append(jac)


		return np.array(jacobians)

	def BundleAdjustmentHomography(self,initialization,m=1e-4):
		
		H_0 = np.random.rand(9*self.num_images)
		lower_bounds = [-np.inf]*(9*self.num_images)
		upper_bounds = [np.inf]*(9*self.num_images)

		for img_name in self.n_to_i_dict:
			i = self.n_to_i_dict[img_name]
			
			H = initialization[img_name]

			H_0[9*i:9*i+9] = H.reshape(9)

		i = self.n_to_i_dict[self.image_reference_name]

		lower_bounds[9*i+0] = 1-m
		upper_bounds[9*i+0] = 1+m

		lower_bounds[9*i+1] = 0-m
		upper_bounds[9*i+1] = 0+m

		lower_bounds[9*i+2] = 0-m
		upper_bounds[9*i+2] = 0+m

		lower_bounds[9*i+3] = 0-m
		upper_bounds[9*i+3] = 0+m

		lower_bounds[9*i+4] = 1-m
		upper_bounds[9*i+4] = 1+m

		lower_bounds[9*i+5] = 0-m
		upper_bounds[9*i+5] = 0+m

		lower_bounds[9*i+6] = 0-m
		upper_bounds[9*i+6] = 0+m

		lower_bounds[9*i+7] = 0-m
		upper_bounds[9*i+7] = 0+m

		lower_bounds[9*i+8] = 1-m
		upper_bounds[9*i+8] = 1+m

		start_time = datetime.datetime.now()

		# Change ftol xtol gtol
		
		res = least_squares(self.get_residuals, H_0,jac=self.get_jacobians, bounds=(lower_bounds,upper_bounds), verbose=2)
		# res = least_squares(self.get_residuals, H_0,jac=self.get_jacobians, verbose=2)
		# res = least_squares(self.get_residuals, H_0,bounds=(lower_bounds,upper_bounds), verbose=2)

		end_time = datetime.datetime.now()
		report_time(start_time,end_time)

		X = res.x

		residuals = res.fun
		# print(X)
		# print(':: Least Squares RMSE: {0}'.format(np.sqrt(np.mean(self.get_residuals(X)))))

		image_corners_dict = {}
		absolute_transformations_dict = {}

		for img_name in self.n_to_i_dict:
			
			i = self.n_to_i_dict[img_name]
			H = X[9*i:9*i+9]
			H = H.reshape(3,3)

			UL_ref = [0,0,1]
			UR_ref = [self.width,0,1]
			LR_ref = [self.width,self.height,1]
			LL_ref = [0,self.height,1]

			UL = np.matmul(H,UL_ref)
			UL = UL/UL[2]
			UL = UL[:2]

			UR = np.matmul(H,UR_ref)
			UR = UR/UR[2]
			UR = UR[:2]

			LR = np.matmul(H,LR_ref)
			LR = LR/LR[2]
			LR = LR[:2]

			LL = np.matmul(H,LL_ref)
			LL = LL/LL[2]
			LL = LL[:2]

			image_corners_dict[img_name] = {'UL':UL,'UR':UR,'LR':LR,'LL':LL}
			absolute_transformations_dict[img_name] = H

		return image_corners_dict,absolute_transformations_dict, end_time-start_time