import numpy as np
from vedo import *
from ipyvtklink.viewer import ViewInteractiveWidget
import time
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from utils import R2Euler, Ell, jacobian_tf, R_tf, get_cluster, get_cluster_fast


class VICET():
	'''
	fid             -> number of azimuthal voxels per 360 deg
	niter           -> number of iterations
	RM              -> Suppress voxels containing moving objects
	max_buffer      -> Max radial buffer on spherical voxels
	mnp             -> Minimum number of points required for a voxel to be considered
	sweep_direction -> VICET assumes LIDAR sensor spins Counter-Clockwise, set to CW otherwise
	'''

	def __init__(self, cloud1, cloud2, fid = 30, niter = 5, draw = True, 
		m_hat0 = np.array([0.0, 0.0, 0., 0., 0., 0.]), chi_init = np.array([0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.]), 
		group = 2, RM = True, DNN_filter = False, cheat = [], mnp = 50, max_buffer = 0.5, sweep_direction = 'CCW'):

		np.random.seed(100)
		tf.random.set_seed(100)

		# self.run_profile = True
		self.run_profile = False
		self.st = time.time() #start time (for debug)

		self.min_cell_distance = 1 #4 #2 #begin closest spherical voxel here
		#ignore "occupied" cells with fewer than this number of pts
		self.min_num_pts = mnp #50 #100 #was 50 for KITTI and Ford, need to lower to 25 for CODD + simulated data
		self.fid = fid # dimension of 3D grid: [fid, fid, fid]
		self.draw = draw
		self.niter = niter
		self.alpha = 0.5 #0.7 #controls alpha values when displaying ellipses
		self.cheat = cheat #overide for using ICET to generate training data for DNN
		self.DNN_filter = DNN_filter
		self.start_filter_iter = 7 #10 #iteration to start DNN rejection filter
		self.start_RM_iter = 8 #iteration to start removing moving objects (set low to generate training data)
		self.DNN_thresh = 0.05 #0.03
		self.RM_thresh = 0.25 #0.05 #0.25
		self.max_buffer = max_buffer #2 max buffer width in spherical voxels

		self.cloud1_tensor = tf.random.shuffle(tf.cast(tf.convert_to_tensor(cloud1), tf.float32))
		#need to reverse order in which points in cloud appear the sensor is spinning CCW
		if sweep_direction == 'CW':
			cloud2 = np.flip(cloud2, axis = 0)
		self.cloud2_tensor = tf.cast(tf.convert_to_tensor(cloud2), tf.float32)

		if self.draw == True:
			self.plt = Plotter(N = 1, axes = 4, bg = (1, 1, 1), interactive = True) #axis = 1
			# self.plt = Plotter(N = 1, axes = 4, bg = (1, 1, 1), interactive = False) #USE FOR MAKING DEMO GIFS 
			self.disp = []
			#copy-paste camera settings here using Shift+C on vedo terminal window
			#follow cam
			self.plt.camera.SetPosition( [-75., 0., 75] ) #0
			self.plt.camera.SetFocalPoint( [15., 0., 2.] ) #0
			self.plt.camera.SetViewUp( [0.65, -0.001, 0.75] )
			self.plt.camera.SetDistance( 115 )
			self.plt.camera.SetClippingRange( [28., 300.] )

		#hold on to cloud1 for dispaying later
		self.cloud1_tensor_OG = self.cloud1_tensor

		#convert cloud to spherical coordinates
		self.cloud1_tensor_spherical = tf.cast(self.c2s(self.cloud1_tensor), tf.float32)
		self.cloud2_tensor_spherical = tf.cast(self.c2s(self.cloud2_tensor), tf.float32)

		#remove  points closer than minimum radial distance
		not_too_close1 = tf.where(self.cloud1_tensor_spherical[:,0] > self.min_cell_distance)[:,0]
		self.cloud1_tensor_spherical = tf.gather(self.cloud1_tensor_spherical, not_too_close1)
		self.cloud1_tensor = tf.gather(self.cloud1_tensor, not_too_close1)

		#DEBUG: remove any elements of cloud that are outside phimin/phimax-----
		phimin =  3*np.pi/8 + 0.01
		# phimin =  1*np.pi/8
		phimax = 7*np.pi/8 - 0.01
		smaller = tf.where(self.cloud1_tensor_spherical[:,2] < phimax)
		bigger = tf.where(self.cloud1_tensor_spherical[:,2] > phimin)
		between = tf.sets.intersection(tf.transpose(smaller),tf.transpose(bigger))
		self.cloud1_tensor = tf.gather(self.cloud1_tensor, between.values)
		self.cloud1_tensor_spherical = tf.gather(self.cloud1_tensor_spherical, between.values)
		#------------------------------------------------------------------------

		not_too_close2 = tf.where(self.cloud2_tensor_spherical[:,0] > self.min_cell_distance)[:,0]
		self.cloud2_tensor_spherical = tf.gather(self.cloud2_tensor_spherical, not_too_close2)
		# self.cloud2_tensor_OG = self.cloud2_tensor #test
		self.cloud2_tensor_OG = tf.gather(self.cloud2_tensor, not_too_close2) #better to remove too close points from OG
		self.cloud2_tensor = tf.gather(self.cloud2_tensor, not_too_close2)

		self.grid_spherical( draw = False )

		self.cloud1_static = None #placeholder for returning inlier points after moving point exclusion routine

		if self.run_profile:
			print("\n converting to spherical took", time.time() - before, "\n total: ",  time.time() - self.st)
			before = time.time()

		print("chi init: \n", chi_init)
		self.solve_12_state(niter = self.niter, A0 = chi_init, remove_moving = RM)

		if self.draw == True:
			#no special camera
			self.plt.show(self.disp, "VICET", resetcam = False) #was this

		# 	#init with fixed camera angle (for making figures)
		# 	cam = dict(
		# 		pos=(-46.32390, -12.33435, 56.92717),
		# 		focalPoint=(2.001187, 0.6357523, -3.620514),
		# 		viewup=(0.7370861, 0.2263244, 0.6367742),
		# 		distance=78.54655,
		# 		clippingRange=(46.50687, 106.3192),
		# 		)
		# 	lb = LegendBox([self.PointsObj1, self.PointsObj2], width=0.3, height=0.2, markers='s', bg = 'white', pos = 'top right', alpha = 0.1).font("Theemim")
		# 	np.set_printoptions(precision=3, suppress=True)
		# 	headerText = "Rigid Transform [x, y, z, roll, pitch, yaw]:  \n" + str(self.A[:6]) +"\n \n Motion Correction [x, y, z, roll, pitch, yaw]: \n" + str(self.A[6:])
		# 	self.plt.show(self.disp, lb, headerText, camera=cam)
		# 	fn = 'figures/gifs/scene2_' + str(self.niter)+ '.png'
		# 	screenshot(fn)

	def solve_12_state(self, niter, A0, remove_moving = False):
		""" Jonitly solve for rigid transformation AND linear motion distortion 
			
			niter = number of iterations to run
			A0  = [rigid transform soln, distortion correction soln] = [X, m]
		"""

		self.A = A0

		# get boundaries containing useful clusters of points from first scan
		thetamin = -np.pi
		thetamax = np.pi
		phimin =  3*np.pi/8
		# phimin =  1*np.pi/8
		phimax = 7*np.pi/8 
		edges_phi = tf.linspace(phimin, phimax, self.fid_phi)
		edges_theta = tf.linspace(thetamin, thetamax, self.fid_theta + 1)

		cloud = self.cloud1_tensor_spherical

		bins_theta = tfp.stats.find_bins(cloud[:,1], edges_theta)
		bins_phi = tfp.stats.find_bins(cloud[:,2], edges_phi)

		#combine bins_theta and bins_phi to get spike bins
		bins_spike = tf.cast(bins_theta*(self.fid_phi-1) + bins_phi, tf.int32)

		#save which spike each point is in to ICET object for further analysis
		self.bins_spike = bins_spike

		#find min point in each occupied spike
		occupied_spikes, idxs = tf.unique(bins_spike)
		temp =  tf.where(bins_spike == occupied_spikes[:,None]) #TODO- there has to be a better way to do this... 
		rag = tf.RaggedTensor.from_value_rowids(temp[:,1], temp[:,0])
		idx_by_rag = tf.gather(cloud[:,0], rag)

		rads = tf.transpose(idx_by_rag.to_tensor()) 
		self.rads = rads
		bounds = get_cluster_fast(rads, thresh = 0.5, mnp = self.min_num_pts, max_buffer = self.max_buffer)

		corn = self.get_corners_cluster(occupied_spikes, bounds)
		inside1, npts1 = self.get_points_in_cluster(self.cloud1_tensor_spherical, occupied_spikes, bounds)	

		self.inside1 = inside1
		self.npts1 = npts1
		self.bounds = bounds
		self.occupied_spikes = occupied_spikes

		mu1, sigma1 = self.fit_gaussian(self.cloud1_tensor, inside1, tf.cast(npts1, tf.float32))

		enough1 = tf.where(npts1 > self.min_num_pts)[:,0]
		mu1_enough = tf.gather(mu1, enough1)
		sigma1_enough = tf.gather(sigma1, enough1)

		U, L = self.get_U_and_L_cluster(sigma1_enough, mu1_enough, occupied_spikes, bounds)

		# if self.draw:
		# 	self.visualize_L(mu1_enough, U, L)
			# self.draw_ell(mu1_enough, sigma1_enough, pc = 1, alpha = self.alpha)
			# self.draw_cell(corn)
			# self.draw_car()

		#main loop
		for i in range(niter):

			print("~~~~~~~~~~~Iteration ", i, "~~~~~~~~~~")

			#decompose A into X_hat (rigid transform) and m_hat (distortion)
			self.X_hat = self.A[:6]
			self.m_hat = self.A[6:]

			# motion correcton -> rigid transform (as in paper) ~~~~~~~~~~~~~~~~~~~~~~~~~

			#apply last estimate of correction to origonal point cloud 2
			self.cloud2_tensor = self.apply_motion_profile(self.cloud2_tensor_OG, self.m_hat)
			#apply last rigid transform
			rot = R_tf(self.X_hat[3:]).numpy()
			trans = self.X_hat[:3]
			self.cloud2_tensor = (self.cloud2_tensor @ rot) + trans
			self.cloud2_tensor = tf.cast(self.cloud2_tensor, tf.float32)

			# rigid transform -> motion correction  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			# #apply last rigid transform
			# rot = R_tf(self.X_hat[3:]).numpy()
			# trans = self.X_hat[:3]
			# self.cloud2_tensor = (self.cloud2_tensor_OG @ rot) + trans
			# self.cloud2_tensor = tf.cast(self.cloud2_tensor, tf.float32)
			# #apply last estimate of correction to original point cloud 2
			# self.cloud2_tensor = self.apply_motion_profile(self.cloud2_tensor, self.m_hat) #, period_lidar = 0.1)
			#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

			#convert back to spherical coordinates
			self.cloud2_tensor_spherical = tf.cast(self.c2s(self.cloud2_tensor), tf.float32)

			#IMPORTANT-- after applying distortion correction, remove points from the beginning of scan 2 with an
			#			 absolute rotation of >= 2pi 
			#			NOTE:  -->  removing beginning pts here since newer college dataset is recorded cw but distortion correction
			#				 			code assumes ccw LIDAR rotation  
			#				   --> LIDAR sensor I simulated in ROS spins ccw so flip sign as needed 
			self.yaw_angs =  self.cloud2_tensor_spherical[:,1].numpy()
			yaw_angs_scaled = (self.yaw_angs + 2*np.pi)%(2*np.pi) #was this
			# yaw_angs_scaled = self.yaw_angs #override test
			#get indices len(yaw_angs_scaled)//8 where yaw_angs_scaled is less than pi  
			problem_idx_beginning = np.argwhere(yaw_angs_scaled[:len(yaw_angs_scaled)//4] < np.pi)
			all_idx = np.linspace(0,len(yaw_angs_scaled)-1, len(yaw_angs_scaled))
			good_idx = np.setdiff1d(all_idx, problem_idx_beginning).astype(np.int32)

			# Also remove points from the end of scan2 with an absolute rotation of < 0 
			# 	(this will happen due to rigid transform)	
			#TODO-- need to mess with indices since we're not starting  at zero here
			yaw_angs_scaled = self.yaw_angs #override test
			problem_idx_end = np.argwhere(yaw_angs_scaled[((3*len(yaw_angs_scaled))//4):] < 0)
			problem_idx_end += (3*len(yaw_angs_scaled))//4
			self.problem_idx_end = problem_idx_end
			good_idx = np.setdiff1d(good_idx, problem_idx_end).astype(np.int32)

			#only hold on to good index points
			self.cloud2_tensor_spherical = tf.gather(self.cloud2_tensor_spherical, good_idx)
			self.yaw_angs_fixed = self.cloud2_tensor_spherical[:,1].numpy()
			self.cloud2_tensor_unshuffled = self.s2c(self.cloud2_tensor_spherical)
			self.cloud2_tensor_spherical = tf.random.shuffle(tf.cast(tf.convert_to_tensor(self.cloud2_tensor_spherical), tf.float32))
			#hold on to a copy in cartesian space with same shuffling
			self.cloud2_tensor = tf.cast(self.s2c(self.cloud2_tensor_spherical), tf.float32)


			#----------------------------------------------------------------------------------------------------

			#find points from scan 2 that fall inside clusters
			inside2, npts2 = self.get_points_in_cluster(self.cloud2_tensor_spherical, occupied_spikes, bounds)
			self.inside2 = inside2
			self.npts2 = npts2

			#fit gaussians distributions to each of these groups of points 		
			mu2, sigma2 = self.fit_gaussian(self.cloud2_tensor, inside2, tf.cast(npts2, tf.float32))

			enough2 = tf.where(npts2 > self.min_num_pts)[:,0]
			mu2_enough = tf.gather(mu2, enough2)

			#get correspondences -- only where there are enough points from both scan 1 and scan 2 in a cell!
			corr = tf.sets.intersection(enough1[None,:], enough2[None,:]).values
			corr_full = tf.sets.intersection(enough1[None,:], enough2[None,:]).values


			#----------------------------------------------
			if remove_moving:  
				if i >= self.start_RM_iter: #TODO: tune this to optimal value
					print("\n ---checking for moving objects---")
					#FIND CELLS THAT INTRODUCE THE MOST ERROR
					
					#test - re-calculting this here
					y0_i_full = tf.gather(mu1, corr_full)
					y_i_full = tf.gather(mu2, corr_full)

					self.residuals_full = y_i_full - y0_i_full
				
					# #------------------------------------------------------------------------------------------------
					# #Using binned mode oulier exclusion (get rid of everything outside of some range close to 0)
					# nbins = 30
					# edges = tf.linspace(-0.75, 0.75, nbins)
					# bins_soln = tfp.stats.find_bins(self.residuals_full[:,0], edges)
					# bad_idx = tf.where(bins_soln != (nbins//2 - 1))[:,0][None, :]

					# bins_soln2 = tfp.stats.find_bins(self.residuals_full[:,1], edges)
					# bad_idx2 = tf.where(bins_soln2 != (nbins//2 - 1))[:,0][None, :]
					# bad_idx = tf.sets.union(bad_idx, bad_idx2).values
					# #------------------------------------------------------------------------------------------------

					# #------------------------------------------------------------------------------------------------
					# #Using Gaussian n-sigma outlier exclusion on translation

					# metric1 = self.residuals_full[:,0]
					# metric2 = self.residuals_full[:,1]
					# mu_x = tf.math.reduce_mean(metric1)
					# sigma_x = tf.math.reduce_std(metric1)
					
					# # #just x------------
					# # bad_idx = tf.where( tf.math.abs(metric1) > mu_x + 2*sigma_x )[:, 0]
					# # #------------------

					# #x and y---------
					# bad_idx = tf.where( tf.math.abs(metric1) > mu_x + 2.0*sigma_x )[:,0][None, :]
					# # print(" \n bad_idx1", bad_idx)

					# mu_y = tf.math.reduce_mean(metric2)
					# sigma_y = tf.math.reduce_std(metric2)
					# bad_idx2 = tf.where( tf.math.abs(metric2) > mu_y + 	2.0*sigma_y )[:,0][None, :]
					# # print("\n bad_idx2", bad_idx2)
					# bad_idx = tf.sets.union(bad_idx, bad_idx2).values

					# #if using rotation too
					# # self.bad_idx = bad_idx
					# # self.bad_idx_rot = bad_idx_rot
					# # bad_idx = tf.sets.union(bad_idx[None, :], bad_idx_rot[None, :]).values 
					# #-----------------

					# # print("corr \n", corr)
					# print("bad idx", bad_idx)
					# # print(tf.gather(it.dx_i[:,0], bad_idx))
					# # print(tf.gather(occupied_spikes, corr))
					# #------------------------------------------------------------------------------------------------

					#hard cutoff for outlier rejection
					#NEW (5/7)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
					both = tf.sets.intersection(enough1[None,:], corr_full[None,:]).values
					#get indices of mu1 that correspond to mu2 that also have sufficient number of points
					ans = tf.where(enough1[:,None] == both)[:,0]
					
					#test moving these here
					U_i = tf.gather(U, ans)
					U_iT = tf.transpose(U_i, [0,2,1])
					L_i = tf.gather(L, ans)
					# residuals_compact = L_i @ U_i @ tf.gather(self.residuals_full[:,:,None], corr_full) #was this (incorrect)
					# residuals_compact = L_i @ U_iT @ tf.gather(self.residuals_full[:,:,None], ans) #(5/19) -> debug: should this be U_i or U_iT?
					residuals_compact = L_i @ U_iT @ self.residuals_full[:,:,None] #correct (5/20)

					# self.RM_thresh = 0.03 #0.1 #0.05
					# bidx = tf.where(residuals_compact > thresh )[:,0] #TODO: consider absolute value!
					bidx = tf.where(tf.math.abs(residuals_compact) > self.RM_thresh )[:,0]
					# print(residuals_compact)
					bad_idx = bidx
					# print("bad_idx", bidx)
					#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

					# #------------------------------------------------------------------------------------------------
					# Compare rotation about the vertical axis between each distribution correspondance
					s1 = tf.transpose(tf.gather(sigma1, corr), [1, 2, 0])
					s2 = tf.transpose(tf.gather(sigma2, corr), [1, 2, 0])

					self.angs1 = R2Euler(s1)[2,:]
					self.angs2 = R2Euler(s2)[2,:]

					self.res = self.angs1 - self.angs2

					mean = np.mean(self.res)
					std = np.std(self.res)
					# bad_idx_rot = tf.where(np.abs(self.res) > mean + 1*std )[:, 0]

					cutoff = 0.1 #0.1
					bad_idx_rot = tf.where(np.abs(self.res) > cutoff)[:, 0]

					# print("bad_idx_rot", bad_idx_rot)

					bad_idx = tf.sets.union(bad_idx[None, :], bad_idx_rot[None, :]).values
					# # #------------------------------------------------------------------------------------------------


					bounds_bad = tf.gather(bounds, tf.gather(corr, bad_idx))
					bad_idx_corn_moving = self.get_corners_cluster(tf.gather(occupied_spikes, tf.gather(corr, bad_idx)), bounds_bad)

					ignore_these = tf.gather(corr, bad_idx)
					corr = tf.sets.difference(corr[None, :], ignore_these[None, :]).values

					#temp
					self.U_i = U_i
					self.L_i = L_i

					# print("\n ~~~~~~~~~~~~~~ \n removed moving", time.time() - before, "\n total: ",  time.time() - self.st, "\n ~~~~~~~~~~~~~~")
					before = time.time()
			#----------------------------------------------


			y_i_full = tf.gather(mu1, corr_full)
			y_j_full = tf.gather(mu2, corr_full)

			y_i = tf.gather(mu1, corr)
			sigma_i = tf.gather(sigma1, corr)
			npts_i = tf.gather(npts1, corr)
			# print(sigma1)

			y_j = tf.gather(mu2, corr)
			sigma_j = tf.gather(sigma2, corr)
			npts_j = tf.gather(npts2, corr)

			#need special indexing for U_i and L_i since they are derived from <mu1_enough>
			# rather than the full mu1 tensor:
			#  1) get IDX of elements that are in both enough1 and corr
			#  2) use this to index U and L to get U_i and L_i
			both = tf.sets.intersection(enough1[None,:], corr[None,:]).values
			ans = tf.where(enough1[:,None] == both)[:,0]			
			U_I = tf.gather(U, ans)
			L_I = tf.gather(L, ans)


			# Get Jacobian H = [H_x, H_m]
			H_m = self.get_H_m(y_j, self.m_hat)
			#remove extra elements of H that we needed for homogenous coordinate transforms
			H_m = tf.reshape(H_m, (tf.shape(H_m)[0]//4,4,6)) 
			H_m = tf.cast(H_m[:,:3,:], tf.float32)

			# shape(H_x) = [num of corr * 3, 6]
			H_x = jacobian_tf(tf.transpose(y_i), tf.cast(self.X_hat[3:], tf.float32)) 
			H_x = tf.reshape(H_x, (tf.shape(H_x)[0]//3, 3, 6)) # -> need shape [#corr//4, 4, 6]
			H = tf.concat([H_x, H_m], axis = 2)
			# print("H.T  \n", H[0,:,:6].numpy().T, "\n", H[0,:,6:].numpy().T)
			# H = tf.concat([H_x, -H_m], axis = 2) #DEBUG

			#construct sensor noise covariance matrix
			R_noise = (tf.transpose(tf.transpose(sigma_i, [1,2,0]) / tf.cast(npts_i - 1, tf.float32)) + 
						tf.transpose(tf.transpose(sigma_j, [1,2,0]) / tf.cast(npts_j - 1, tf.float32)) )
			# print("R_noise:", np.shape(R_noise))

			#use projection matrix to remove extended directions
			R_noise = L_I @ tf.transpose(U_I, [0,2,1]) @ R_noise @ U_I @ tf.transpose(L_I, [0,2,1])
			#take inverse of R_noise to get our weighting matrix
			W = tf.linalg.pinv(R_noise)

			# use LUT to remove rows of H corresponding to overly extended directions
			LUT = L_I @ tf.transpose(U_I, [0,2,1])
			# print("LUT", tf.shape(LUT))
			# H_z = LUT @ H
			H_z = H #-- taking care of extended surfaces in a minute

			HTWH = tf.math.reduce_sum(tf.matmul(tf.matmul(tf.transpose(H_z, [0,2,1]), W), H_z), axis = 0) #was this for ICET 
			# HTWH = tf.matmul(tf.matmul(tf.transpose(H_z, [0,2,1]), W), H_z) #need to hold off on summing until the end
			# HTWH = tf.matmul(tf.transpose(H_z, [0,2,1]), H_z) #test-- ignore weighting for now??
			HTW = tf.matmul(tf.transpose(H_z, [0,2,1]), W)
			# HTW = tf.math.reduce_sum(tf.matmul(tf.transpose(H_z, [0,2,1]), W), axis = 0) #wrong-- need to apply to residual vec before summing... 

			#get output covariance matrix
			self.Q = tf.linalg.pinv(HTWH)
			self.pred_stds = tf.linalg.tensor_diag_part(tf.math.sqrt(tf.abs(self.Q)))

			#need to get (H.T W) to shape [12, 3N], where N is the number of correspondnces
			HTW = tf.transpose(HTW, [1,0,2]) #need to get things in the correct order for the reshape operation???
			HTW = tf.reshape(HTW, [12,-1])			
			# print("HTWH \n", np.shape(HTWH))
			# print("HTW \n", np.shape(HTW), HTW[:30])
			
			# compact residuals: ~~~~~~~~~~~~~~~~~
			# print("(y_i -  y_j)", np.shape(y_i -  y_j))
			# residuals_compact = tf.reshape((U_I @ L_I @ tf.transpose(U_I, [0,2,1])), [-1,3] ) @ (y_i -  y_j).numpy().flatten() 
			# print("\n residuals_compact", np.shape(residuals_compact))
			temp = ((U_I @ L_I @ tf.transpose(U_I, [0,2,1])) @ (y_j - y_i)[:,:,None])[:,:,0]
			# print("temp", np.shape(temp), "\n", temp[:10])
			residuals_compact = temp.numpy().flatten()
			# print(residuals_compact)
			# print("y_i", np.shape(y_i))

			# #DEBUG: draw arrows
			# for count in range(len(y_i)):
			# 	compact_residual_arrows = shapes.Arrow(y_i[count].numpy(), y_i[count].numpy() + temp[count], c = "red")
			# 	self.disp.append(compact_residual_arrows)
			# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

			# using full residuals
			# residuals = (y_j -  y_i).numpy().flatten()[:,None] #was this
			residuals = residuals_compact[:,None] #works way better! Just remember to suppress H_z with H
			# print("\n residuals", np.shape(residuals))
			delta_A =  (tf.linalg.pinv(HTWH) @ HTW @ residuals)[:,0]
			# print("\n delta_A\n", np.shape(delta_A))

			# #DEBUG~~~
			# # residuals = tf.reshape((y_i -  y_j), [-1,1])
			# residuals = (y_i -  y_j)
			# print("\n residuals", np.shape(residuals))
			# print("\n  (HTWH)-1 HT \n", np.shape(tf.reshape((tf.linalg.pinv(HTWH) @ HTW), [-1,3])))
			# delta_A =  tf.reshape((tf.linalg.pinv(HTWH) @ HTW), [-1,3]) @ residuals #test
			# print("\n delta_A before \n", np.shape(delta_A))

			#need to sum up all contributions
			# delta_A = tf.math.reduce_sum(delta_A, axis = 0)#[:,0]
			# print("\n delta_A \n", np.round(delta_A, 3)[:6], "\n", np.round(delta_A, 3)[6:])

			# #apply both at once
			# # #augment rigid transform components
			# self.A[:3] += delta_A[:3]
			# self.A[3:6] += delta_A[3:6]
			# #augment distortion correction
			# self.A[6:9] += delta_A[6:9]
			# self.A[9:] += delta_A[9:] #sign was flipped

			#Scale down-- seems to work better on more challenging scenes
			# rigid transform components
			self.A[:3] += 0.2*delta_A[:3]
			self.A[3:6] += 0.2*delta_A[3:6]
			# distortion correction
			self.A[6:9] += 0.2*delta_A[6:9]
			self.A[9:] += 0.2*delta_A[9:]

			print("chi hat: \n", np.round(self.A, 4)[:6], "\n", np.round(self.A, 4)[6:])

		if self.draw:
			self.draw_cloud(self.cloud1_tensor, pc = 1) #show only what fits inside grid
			# self.draw_cloud(self.cloud2_tensor_OG, pc = 1) 
			self.draw_cloud(self.cloud2_tensor, pc = 2)

			# self.disp.append(Points(self.cloud1_tensor_OG, c='red',  r = 3.5, alpha =0.2))  
			# self.disp.append(Points(self.cloud2_tensor_OG, c='blue',  r = 3, alpha =0.5))

			# color = 255*np.linspace(0,1,len(self.cloud2_tensor))
			# cname = np.array([255-color, color, 255-color]).T.tolist()
			# self.disp.append(Points(self.cloud2_tensor_unshuffled, c=cname,  r = 3.5, alpha =0.5))

		# 	# self.draw_ell(y_j, sigma_j, pc = 2, alpha = self.alpha)
		# 	# self.draw_ell(y_i, sigma_i, pc = 1, alpha = self.alpha)

		# 	if remove_moving:
		# 		self.draw_cell(bad_idx_corn_moving, bad = True)
		# 	# self.draw_correspondences(mu1, mu2, corr) #corr displays just used correspondences


	def apply_motion_profile(self, cloud_xyz, m_hat, period_lidar = 1):
		"""Linear correction for motion distortion, using ugly python code

		cloud_xyz: distorted cloud in Cartesian space
		m_hat: estimated motion profile for linear correction
		period_lidar: time it takes for LIDAR sensor to record a single sweep
		"""

		period_base = (2*np.pi)/m_hat[-1]

		#remove inf values
		cloud_xyz = cloud_xyz[cloud_xyz[:,0] < 10_000]
		#convert to spherical coordinates
		# cloud_spherical = c2s(cloud_xyz).numpy()

		#Because of body frame yaw rotation, we're not always doing a full roation - we need to "uncurl" initial point cloud
		# (this is NOT baked in to motion profile)
		# cloud_spherical = cloud_spherical[np.argsort(cloud_spherical[:,1])] #sort by azim angle
		#get total overlap in rotation between LIDAR and base frames (since both are rotating w.r.t. world Z)
		# point of intersection = (t_intersection) * (angular velocity base)
		#						= ((n * T_a * T_b) / (T_a + T_b)) * omega_base 
		# total_rot = -2*np.pi*np.sin(m_hat[-1]/(-m_hat[-1] + (2*np.pi/period_lidar)))
		# print("total_rot:", total_rot)

		#scale linearly starting at theta = 0
		# cloud_spherical[:,1] = ((cloud_spherical[:,1]) % (2*np.pi))*((2*np.pi - total_rot)/(2*np.pi)) + total_rot #works

		#sort by azim angle again- some points will have moved past origin in the "uncurling" process
		# cloud_spherical = cloud_spherical[np.argsort(cloud_spherical[:,1])] 

		#reorient
		# cloud_spherical[:,1] = ((cloud_spherical[:,1] + np.pi) % (2*np.pi)) - np.pi
		# cloud_xyz = s2c(cloud_spherical).numpy() #convert back to xyz

		rectified_vel  = -m_hat[None,:]
		# rectified_vel[0,-1] = 0 #zero out yaw since we already compensated for it

		T = (2*np.pi)/(-m_hat[-1] + (2*np.pi/period_lidar)) #time to complete 1 scan #was this
		# print(T)
		rectified_vel = rectified_vel * T #was this
		# print(rectified_vel[:,-1])
		# rectified_vel[:-1] = rectified_vel[:-1] * T #nope
		# rectified_vel[:,-1] = rectified_vel[:,-1] * T #also nope

		# linearly spaced motion profile ~~~~~~~~~~~~~~~~~~~~~
		# this is a bad way of doing it ... what happens if most of the points are on one half of the scene??

		# part2 = np.linspace(0.5, 1.0, len(cloud_xyz)//2)[:,None]
		# part1 = np.linspace(0, 0.5, len(cloud_xyz) - len(cloud_xyz)//2)[:,None]
		# motion_profile = np.append(part1, part2, axis = 0) @ rectified_vel

		# # using yaw angles ~~~~~~~~~~~~~~~~~~~~~~
		#  (NEW)
		
		#TODO: need to center point cloud before getting yaw angles

		yaw_angs = self.c2s(cloud_xyz)[:,1].numpy()
		last_subzero_idx = int(len(yaw_angs) // 8)
		yaw_angs[last_subzero_idx:][yaw_angs[last_subzero_idx:] < 0.3] = yaw_angs[last_subzero_idx:][yaw_angs[last_subzero_idx:] < 0.3] + 2*np.pi
		# yaw_angs = yaw_angs / (2*np.pi) #test
		# yaw_angs = yaw_angs[:,None]  #test

		# #test
		# self.yaw_angs = yaw_angs
		# self.cloud_xyz = cloud_xyz
		# color = 255*self.yaw_angs/(2*np.pi)
		# cname = np.array([255-color, color, 255-color]).T.tolist()
		# # self.disp.append(Points(self.cloud_xyz + 0.01*np.random.randn(np.shape(cloud_xyz)[0],3), c=cname, r = 3))
		# self.disp.append(Points(self.cloud_xyz, c=cname, r = 3))

		#jump in <yaw_angs> is causing unintended behavior in real world LIDAR data
		yaw_angs = (yaw_angs + 2*np.pi)%(2*np.pi)

		#TODO: should I use (2pi - T) in place of max(yaw_angs) -> ???
		motion_profile = (yaw_angs / np.max(yaw_angs))[:,None] @ rectified_vel #was this
		# motion_profile = yaw_angs[:,None] @ rectified_vel #test
		# print("\n new: \n", motion_profile[:,0])
		self.yaw_angs_new = yaw_angs
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


		#Apply motion profile
		# # Old loopy method ~~~~~~~~~~~~~~
		# T = []
		# for i in range(len(motion_profile)):
		# 	tx, ty, tz, roll, pitch, yaw = motion_profile[i]
		# 	R = np.dot(np.dot(np.array([[1, 0, 0], 
		# 								[0, np.cos(roll), -np.sin(roll)], 
		# 								[0, np.sin(roll), np.cos(roll)]]), 
		# 					np.array([[np.cos(pitch), 0, np.sin(pitch)], 
		# 							  [0, 1, 0], 
		# 							  [-np.sin(pitch), 0, np.cos(pitch)]])), 
		# 					np.array([[np.cos(yaw), -np.sin(yaw), 0], 
		# 							  [np.sin(yaw), np.cos(yaw), 0], 
		# 							  [0, 0, 1]]))
		# 	T.append(np.concatenate((np.concatenate((R, np.array([[tx], [ty], [tz]])), axis=1), np.array([[0, 0, 0, 1]])), axis=0))
		
		# #should be the same size:
		# # print(len(T))
		# # Apply inverse of motion transformation to each point
		# undistorted_pc = np.zeros_like(cloud_xyz)
		# for i in range(len(cloud_xyz)):
		# 	point = np.concatenate((cloud_xyz[i], np.array([1])))
		# 	T_inv = np.linalg.inv(T[i])
		# 	corrected_point = np.dot(T_inv, point)[:3]
		# 	undistorted_pc[i] = corrected_point
		# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		# new method ~~~~~~~~~~~~~~~~~~~~~~

		x = -motion_profile[:,0]
		y = -motion_profile[:,1]
		z = -motion_profile[:,2]
		phi = motion_profile[:,3]
		theta = motion_profile[:,4]
		psi = motion_profile[:,5]

		#need to inverse this
		# T_rect_numpy = np.array([[cos(psi)*cos(theta), sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi), sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi), -x], [sin(psi)*cos(theta), sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi), -sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi), -y], [-sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta), -z], [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.ones(len(x))]])
		#to this
		T_rect_numpy = np.array([[cos(psi)*cos(theta), sin(psi)*cos(theta), -sin(theta), x*cos(psi)*cos(theta) + y*sin(psi)*cos(theta) - z*sin(theta)], [sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi), sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi), sin(phi)*cos(theta), x*(sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi)) + y*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi)) + z*sin(phi)*cos(theta)], [sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi), -sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi), cos(phi)*cos(theta), x*(sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi)) - y*(sin(phi)*cos(psi) - sin(psi)*sin(theta)*cos(phi)) + z*cos(phi)*cos(theta)], [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.ones(len(x))]])
		T_rect_numpy = np.transpose(T_rect_numpy, (2,0,1))
		# print(np.shape(cloud_xyz))
		cloud_homo = np.append(cloud_xyz, np.ones([len(cloud_xyz),1]), axis = 1)
		# print("cloud homo", np.shape(cloud_homo))

		# undistorted_pc =  (np.linalg.pinv(T_rect_numpy) @ cloud_homo[:,:,None]).astype(np.float32)
		undistorted_pc =  (T_rect_numpy @ cloud_homo[:,:,None]).astype(np.float32)
		# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


		return undistorted_pc[:,:3,0]


	def get_H_m(self, y_j, m_hat):
		"""calculate appended Jacobian matrices H wrt m_hat
		
			y_j: [N, 4] list of distribution cecnters in cartesian (x, y, z) 
						or homogenous coordinates (x, y, z, 1)
			m_hat: estimated acceleration [x, y, z, phi, theta, psi] 

		"""

		if np.shape(y_j)[1] == 3:
			y_j = np.append(y_j, np.ones([len(y_j),1]), axis = 1)
		# print(y_j)

		#get motion profile M using m_hat and lidar command velocity

		# #old way-- assumes full coverage of point returns (not always the case) ~~
		## get scaling time (for composite yaw rotation)
		# period_lidar = 1
		# t_scale = (2*np.pi)/(-m_hat[-1] + (2*np.pi/period_lidar))
		# svec = np.linspace(0,t_scale, len(y_j))
		# M = m_hat * np.array([svec, svec, svec, svec, svec, svec]).T

		# new way-- scale M by azimuth angle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		yaw_angs = self.c2s(y_j)[:,1].numpy()
		last_subzero_idx = int(len(yaw_angs) // 8)
		yaw_angs[last_subzero_idx:][yaw_angs[last_subzero_idx:] < 0.3] = yaw_angs[last_subzero_idx:][yaw_angs[last_subzero_idx:] < 0.3] + 2*np.pi

		# print("yaw_angs", len(yaw_angs))

		svec = (yaw_angs / np.max(yaw_angs))  #scaling vec
		# print("\n svec: \n", np.shape(svec), svec)
		M = m_hat * np.array([svec, svec, svec, svec, svec, svec]).T
		# print("\n M \n", np.shape(M))
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


		#construct H matrix
		from numpy import sin, cos

		#decompose M into constituant vectors x, y, z, phi, theta, psi
		x = M[:,0]
		y = M[:,1]
		z = M[:,2]
		phi = M[:,3]
		theta = M[:,4]
		psi = M[:,5]

		#print output of analytic derivatives to create numpy matrices, run everything in vector form
		dT_rect_dx = np.array([[np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), -np.ones(len(x))],
							   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))],
							   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))],
							   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))]])
		dT_rect_dx = np.transpose(dT_rect_dx, (2,0,1)) #reorder to (N, 4, 4)  
		H_x = dT_rect_dx @ y_j[:,:,None] #multiply by vector of points under consideration
		H_x = np.reshape(H_x, (-1,1)) #reshape to (4N x 1)

		dT_rect_dy = np.array([[np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))],
							   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), -np.ones(len(x))],
							   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))],
							   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))]])
		dT_rect_dy = np.transpose(dT_rect_dy, (2,0,1)) #reorder to (N, 4, 4)  
		H_y = dT_rect_dy @ y_j[:,:,None] #multiply by vector of points under consideration
		H_y = np.reshape(H_y, (-1,1)) #reshape to (4N x 1)

		dT_rect_dz = np.array([[np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))],
							   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))],
							   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), -np.ones(len(x))],
							   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))]])
		dT_rect_dz = np.transpose(dT_rect_dz, (2,0,1)) #reorder to (N, 4, 4)  
		H_z = dT_rect_dz @ y_j[:,:,None] #multiply by vector of points under consideration
		H_z = np.reshape(H_z, (-1,1)) #reshape to (4N x 1)

		dT_rect_dphi = np.array([[np.zeros(len(x)), (sin(phi)*sin(psi)*sin(theta)**2 + sin(phi)*sin(psi)*cos(theta)**2 + sin(theta)*cos(phi)*cos(psi))/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), (-sin(phi)*sin(theta)*cos(psi) + sin(psi)*sin(theta)**2*cos(phi) + sin(psi)*cos(phi)*cos(theta)**2)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), np.zeros(len(x))], [np.zeros(len(x)), (-sin(phi)*sin(theta)**2*cos(psi) - sin(phi)*cos(psi)*cos(theta)**2 + sin(psi)*sin(theta)*cos(phi))/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), (-sin(phi)*sin(psi)*sin(theta) - sin(theta)**2*cos(phi)*cos(psi) - cos(phi)*cos(psi)*cos(theta)**2)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), np.zeros(len(x))], [np.zeros(len(x)), cos(phi)*cos(theta)/(sin(phi)**2*sin(theta)**2 + sin(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2 + cos(phi)**2*cos(theta)**2), -sin(phi)*cos(theta)/(sin(phi)**2*sin(theta)**2 + sin(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2 + cos(phi)**2*cos(theta)**2), np.zeros(len(x))], [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))]])
		dT_rect_dphi = np.transpose(dT_rect_dphi, (2,0,1)) #reorder to (N, 4, 4)  
		H_phi = dT_rect_dphi @ y_j[:,:,None] #multiply by vector of points under consideration
		H_phi = np.reshape(H_phi, (-1,1)) #reshape to (4N x 1)

		dT_rect_dtheta = np.array([[-sin(theta)*cos(psi)/(sin(psi)**2*sin(theta)**2 + sin(psi)**2*cos(theta)**2 + sin(theta)**2*cos(psi)**2 + cos(psi)**2*cos(theta)**2), sin(phi)*cos(psi)*cos(theta)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), cos(phi)*cos(psi)*cos(theta)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), np.zeros(len(x))], [-sin(psi)*sin(theta)/(sin(psi)**2*sin(theta)**2 + sin(psi)**2*cos(theta)**2 + sin(theta)**2*cos(psi)**2 + cos(psi)**2*cos(theta)**2), sin(phi)*sin(psi)*cos(theta)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), sin(psi)*cos(phi)*cos(theta)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), np.zeros(len(x))], [-cos(theta)/(sin(theta)**2 + cos(theta)**2), -sin(phi)*sin(theta)/(sin(phi)**2*sin(theta)**2 + sin(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2 + cos(phi)**2*cos(theta)**2), -sin(theta)*cos(phi)/(sin(phi)**2*sin(theta)**2 + sin(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2 + cos(phi)**2*cos(theta)**2), np.zeros(len(x))], [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))]])
		dT_rect_dtheta = np.transpose(dT_rect_dtheta, (2,0,1)) #reorder to (N, 4, 4)  
		H_theta = dT_rect_dtheta @ y_j[:,:,None] #multiply by vector of points under consideration
		H_theta = np.reshape(H_theta, (-1,1)) #reshape to (4N x 1)

		dT_rect_dpsi = np.array([[-sin(psi)*cos(theta)/(sin(psi)**2*sin(theta)**2 + sin(psi)**2*cos(theta)**2 + sin(theta)**2*cos(psi)**2 + cos(psi)**2*cos(theta)**2), (-sin(phi)*sin(psi)*sin(theta) - sin(theta)**2*cos(phi)*cos(psi) - cos(phi)*cos(psi)*cos(theta)**2)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), (sin(phi)*sin(theta)**2*cos(psi) + sin(phi)*cos(psi)*cos(theta)**2 - sin(psi)*sin(theta)*cos(phi))/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), np.zeros(len(x))], [cos(psi)*cos(theta)/(sin(psi)**2*sin(theta)**2 + sin(psi)**2*cos(theta)**2 + sin(theta)**2*cos(psi)**2 + cos(psi)**2*cos(theta)**2), (sin(phi)*sin(theta)*cos(psi) - sin(psi)*sin(theta)**2*cos(phi) - sin(psi)*cos(phi)*cos(theta)**2)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), (sin(phi)*sin(psi)*sin(theta)**2 + sin(phi)*sin(psi)*cos(theta)**2 + sin(theta)*cos(phi)*cos(psi))/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), np.zeros(len(x))], [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))], [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))]])
		dT_rect_dpsi = np.transpose(dT_rect_dpsi, (2,0,1)) #reorder to (N, 4, 4)  
		H_psi = dT_rect_dpsi @ y_j[:,:,None] #multiply by vector of points under consideration
		H_psi = np.reshape(H_psi, (-1,1)) #reshape to (4N x 1)

		# H = np.array([H_x, H_y, H_z, H_phi, H_theta, H_psi]).T[0] #was this
		H = np.array([H_x, H_y, H_z, -H_phi, -H_theta, -H_psi]).T[0] #seems to work better- so don't need to flip sign of delta_A for rotation states later on

		# print(H_phi)

		#scale each element of H_m proportional to theta angle
		H_m = H * np.repeat(np.array([svec, svec, svec, svec, svec, svec]).T, repeats = 4, axis = 0)

		# print("H_m \n", np.shape(H_m))

		return H_m

	def shade_residuals(self, corners, residuals):
		""" draw cells in frame, shaded by how large their residuals are """

		# print("residuals:", tf.shape(self.residuals)) 
		# print("corn:", tf.shape(self.corn))

		#convert residuals to L2 distance 
		L2 = tf.sqrt(tf.reduce_sum(residuals**2, axis = 1))
		L2 = L2.numpy()
		# L2 = tf.sqrt(L2).numpy() #make differences less exaggerated (for viz)
		cap = self.RM_thresh #0.15 #(meters)
		L2[L2 > cap] = cap
		#scale relative to max L2
		biggest_residual = tf.math.reduce_max(L2).numpy()
		print("biggest_residual:", biggest_residual)
		L2 = L2/biggest_residual
		# print(L2)

		for i in range(tf.shape(corners)[0]):

			p1, p2, p3, p4, p5, p6, p7, p8 = self.s2c(corners[i]).numpy()

			lineWidth = 2.5
			a = 0.65 #alpha
			# c1 = 'black'
			c1 = np.array([L2[i], 1-L2[i], 0.2]) #green -> red
			# c1 = np.array([1-L2[i], 1-L2[i], 1-L2[i]])

			# arc1 = shapes.Arc(center = [0,0,0], point1 = p1, point2 = p2, c = 'red')	
			arc1 = shapes.Line(p1, p2, c = c1, lw = lineWidth, alpha = a) #debug		
			self.disp.append(arc1)
			# arc2 = shapes.Arc(center = [0,0,0], point1 = p3, point2 = p4, c = 'red')
			arc2 = shapes.Line(p3, p4, c = c1, lw = lineWidth, alpha = a) #debug
			self.disp.append(arc2)
			line1 = shapes.Line(p1, p3, c = c1, lw = lineWidth, alpha = a)
			self.disp.append(line1)
			line2 = shapes.Line(p2, p4, c = c1, lw = lineWidth, alpha = a) #problem here
			self.disp.append(line2)

			# arc3 = shapes.Arc(center = [0,0,0], point1 = p5, point2 = p6, c = 'red')		
			arc3 = shapes.Line(p5, p6, c = c1, lw = lineWidth, alpha = a) #debug
			self.disp.append(arc3)
			# arc4 = shapes.Arc(center = [0,0,0], point1 = p7, point2 = p8, c = 'red')
			arc4 = shapes.Line(p7, p8, c = c1, lw = lineWidth, alpha = a) #debug
			self.disp.append(arc4)
			line3 = shapes.Line(p5, p7, c = c1, lw = lineWidth, alpha = a)
			self.disp.append(line3)
			line4 = shapes.Line(p6, p8, c = c1, lw = lineWidth, alpha = a)
			self.disp.append(line4)

			self.disp.append(shapes.Line(p1,p5, c = c1, lw = lineWidth, alpha = a))
			self.disp.append(shapes.Line(p2,p6, c = c1, lw = lineWidth, alpha = a))
			self.disp.append(shapes.Line(p3,p7, c = c1, lw = lineWidth, alpha = a))
			self.disp.append(shapes.Line(p4,p8, c = c1, lw = lineWidth, alpha = a))


	def get_points_in_cluster(self, cloud, occupied_spikes, bounds):
		""" returns ragged tensor containing the indices of points in <cloud> in each cluster 
		
			cloud = point cloud tensor
			occupied_spikes = tensor containing idx of spikes corresponding to bounds
			bounds = tensor containing min and max radius for each occupied spike

			#TODO: this is SUPER inefficient rn- 

		"""
		# st = time.time()

		thetamin = -np.pi
		thetamax = np.pi #-  2*np.pi/self.fid_theta
		phimin =  3*np.pi/8
		# phimin =  1*np.pi/8
		phimax = 7*np.pi/8

		edges_phi = tf.linspace(phimin, phimax, self.fid_phi) #was this for regular cells
		bins_phi = tfp.stats.find_bins(cloud[:,2], edges_phi)

		edges_theta = tf.linspace(thetamin, thetamax, self.fid_theta + 1)
		bins_theta = tfp.stats.find_bins(cloud[:,1], edges_theta)

		spike_idx = tf.cast(bins_theta*(self.fid_phi-1) + bins_phi, tf.int32)

		#TODO- bottleneck here, try using equation to ID spikes each point, then consider radial bins...

		#get idx of spike for each applicable point
		cond1 = spike_idx == occupied_spikes[:,None] #match spike IDs
		cond2 = cloud[:,0] < tf.cast(bounds[:,1][:,None], tf.float32) #closer than max bound
		cond3 = cloud[:,0] > tf.cast(bounds[:,0][:,None], tf.float32) #further than min bound
		# cond1 = tf.math.equal(spike_idx, occupied_spikes[:,None]) #match spike IDs
		# cond2 = tf.math.less(cloud[:,0], bounds[:,1][:,None]) #closer than max bound
		# cond3 = tf.math.greater(cloud[:,0], bounds[:,0][:,None]) #further than min bound

		inside1 = tf.where(tf.math.reduce_all(tf.Variable([cond1, cond2, cond3]), axis = 0))
		numPtsPerCluster = tf.math.bincount(tf.cast(inside1[:,0], tf.int32))
		inside1 = tf.RaggedTensor.from_value_rowids(inside1[:,1], inside1[:,0])

		# print("\n took ", time.time() -st , "seconds to get points in cluster" )

		return(inside1, numPtsPerCluster)

	def get_U_and_L_cluster(self, sigma1, mu1, occupied_spikes, bounds):
		""" get U and L when using cluster point grouping """ 


		eigenval, eigenvec = tf.linalg.eig(sigma1)
		U = tf.math.real(eigenvec) #was this
		# U = tf.transpose(tf.math.real(eigenvec), [0, 2, 1]) #wrong! (but looks right if we are drawing arrows incorrectly)

		# print("eigenval", tf.math.real(eigenval))

		#need to create [N,3,3] diagonal matrices for axislens
		zeros = tf.zeros([tf.shape(tf.math.real(eigenval))[0]])
		axislen = tf.Variable([tf.math.real(eigenval)[:,0], zeros, zeros,
							   zeros, tf.math.real(eigenval)[:,1], zeros,
							   zeros, zeros, tf.math.real(eigenval)[:,2]])

		axislen = tf.reshape(tf.transpose(axislen), (tf.shape(axislen)[1], 3, 3)) #variance not std...?

		# get projections of axis length in each direction
		rotated = tf.matmul(axislen, tf.transpose(U, [0, 2, 1])) #new

		# axislen_actual = 2*tf.math.sqrt(axislen) #theoretically correct
		axislen_actual = 3*tf.math.sqrt(axislen) #was this (works with one edge extended detection criteria)
		# axislen_actual = 0.1*tf.math.sqrt(axislen) #turns off extended axis pruning
		# print(axislen_actual)
		rotated_actual = tf.matmul(axislen_actual, tf.transpose(U, [0, 2, 1]))
		# print("rotated_actual", rotated_actual)
	
		#get points at the ends of each distribution ellipse
		# print("mu1", mu1)
		mu_repeated = tf.tile(mu1, [3,1])
		mu_repeated = tf.reshape(tf.transpose(mu_repeated), [3,3,-1])
		mu_repeated = tf.transpose(mu_repeated, [2,1,0])
		mu_repeated = tf.reshape(mu_repeated, [-1,3, 3])
		# print("mu_repeated", mu_repeated)

		P1 = mu_repeated + rotated_actual
		P1 = tf.reshape(P1, [-1, 3])
		P2 = mu_repeated - rotated_actual
		P2 = tf.reshape(P2, [-1, 3])

		#Assumes mu is always going to be inside the corresponding cell (should almost always be the case, if not, its going to fail anyways)
		insideP_ideal, nptsP_ideal = self.get_points_in_cluster(self.c2s(tf.reshape(mu_repeated, [-1,3])), occupied_spikes, bounds)
		insideP_ideal = insideP_ideal.to_tensor()
		# print("insideP_ideal", insideP_ideal)


		#find which points in P are actually inside which cell in <cells>
		insideP1_actual, nptsP1_actual = self.get_points_in_cluster(self.c2s(P1), occupied_spikes, bounds)
		# print(insideP1_actual)
		insideP1_actual = insideP1_actual.to_tensor(shape = tf.shape(insideP_ideal)) #force to be same size as insideP_ideal
		# print("insideP1_actual", insideP1_actual)
		insideP2_actual, nptsP2_actual = self.get_points_in_cluster(self.c2s(P2), occupied_spikes, bounds)
		insideP2_actual = insideP2_actual.to_tensor(shape = tf.shape(insideP_ideal))

		#compare the points inside each cell to how many there are supposed to be
		#	(any mismatch signifies an overly extended direction)
		bofa1 = tf.sets.intersection(insideP_ideal, insideP1_actual).values
		# print("\n bofa1", bofa1)
		bofa2 = tf.sets.intersection(insideP_ideal, insideP2_actual).values
		# print("\n bofa2", bofa2)

		#combine both the positive and negative axis directions
		# deez = tf.cast(tf.sets.intersection(bofa1[None, :], bofa2[None, :]).values[:,None], tf.int32) # only need one edge outside cell (was this)
		deez = tf.cast(tf.sets.union(bofa1[None, :], bofa2[None, :]).values[:,None], tf.int32) #both edeges need to be outside cell to be ambigous
		# print("unambiguous indices", deez)

		data = tf.ones((tf.shape(deez)[0],3))
		I = tf.tile(tf.eye(3), (tf.shape(U)[0], 1))

		mask = tf.scatter_nd(indices = deez, updates = data, shape = tf.shape(I))

		L = mask * I
		L = tf.reshape(L, (tf.shape(L)[0]//3,3,3))

		return(U, L)

			
	def get_U_and_L(self, sigma1, mu1, cells, method = 0):
		""" 	sigma1 = sigmas from the first scan
				cells = tensor containing the indices of each scan
				
				method == 0: old method simiar to 3D-ICET
				method == 1: New "unsceneted KF" strategy

				U = rotation matrix for each voxel to transform scan 2 distribution
				 into frame corresponding to ellipsoid axis in keyframe
			    L = matrix to prune extended directions in each voxel (from keyframe)
			    """

		eigenval, eigenvec = tf.linalg.eig(sigma1)
		U = tf.math.real(eigenvec)

		#need to create [N,3,3] diagonal matrices for axislens
		zeros = tf.zeros([tf.shape(tf.math.real(eigenval))[0]])
		axislen = tf.Variable([tf.math.real(eigenval)[:,0], zeros, zeros,
							   zeros, tf.math.real(eigenval)[:,1], zeros,
							   zeros, zeros, tf.math.real(eigenval)[:,2]])

		axislen = tf.reshape(tf.transpose(axislen), (tf.shape(axislen)[1], 3, 3)) #variance not std...?
		# print("\n axislen \n", axislen)

		#new method (UKF-type strategy)
		#_______________________________________________________________________________
		if method == 1:
			# get projections of axis length in each direction
			rotated = tf.matmul(axislen, tf.transpose(U, [0, 2, 1])) #new

			#QUESTION: should I scale this up if we use stretched voxels?
			axislen_actual = 2*tf.math.sqrt(axislen)
			# print(axislen_actual)
			rotated_actual = tf.matmul(axislen_actual, tf.transpose(U, [0, 2, 1]))
			# print("rotated_actual", rotated_actual)
		
			#get points at the ends of each distribution ellipse
			# print("mu1", mu1)
			mu_repeated = tf.tile(mu1, [3,1])
			mu_repeated = tf.reshape(tf.transpose(mu_repeated), [3,3,-1])
			mu_repeated = tf.transpose(mu_repeated, [2,1,0])
			mu_repeated = tf.reshape(mu_repeated, [-1,3, 3])
			# print("mu_repeated", mu_repeated)

			P1 = mu_repeated + rotated_actual
			P1 = tf.reshape(P1, [-1, 3])
			P2 = mu_repeated - rotated_actual
			P2 = tf.reshape(P2, [-1, 3])

			#draw tempoary marking at boundaries of each distribution ellipse
			# self.disp.append(Points(P1.numpy(), 'g', r = 10))
			# self.disp.append(Points(P2.numpy(), 'g', r = 10))

			#find out which points in P SHOULD be inside each cell
			insideP_ideal, nptsP_ideal = self.get_points_inside(self.c2s(tf.reshape(mu_repeated, [-1,3])), cells[:,None])
			insideP_ideal = insideP_ideal.to_tensor()
			# print("insideP_ideal", insideP_ideal)

			#find which points in P are actually inside which cell in <cells>
			insideP1_actual, nptsP1_actual = self.get_points_inside(self.c2s(P1), cells[:, None])
			# print(insideP1_actual)
			insideP1_actual = insideP1_actual.to_tensor(shape = tf.shape(insideP_ideal)) #force to be same size as insideP_ideal
			# print("insideP1_actual", insideP1_actual)
			insideP2_actual, nptsP2_actual = self.get_points_inside(self.c2s(P2), cells[:, None])
			insideP2_actual = insideP2_actual.to_tensor(shape = tf.shape(insideP_ideal))
			# print("insideP2_actual", insideP2_actual)

			#compare the points inside each cell to how many there are supposed to be
			#	(any mismatch signifies an overly extended direction)
			bofa1 = tf.sets.intersection(insideP_ideal, insideP1_actual).values
			# print("\n bofa1", bofa1)
			bofa2 = tf.sets.intersection(insideP_ideal, insideP2_actual).values
			# print("\n bofa2", bofa2)

			#combine both the positive and negative axis directions
			deez = tf.cast(tf.sets.intersection(bofa1[None, :], bofa2[None, :]).values[:,None], tf.int32)
			# print("unambiguous indices", deez)

			data = tf.ones((tf.shape(deez)[0],3))
			I = tf.tile(tf.eye(3), (tf.shape(U)[0], 1))

			mask = tf.scatter_nd(indices = deez, updates = data, shape = tf.shape(I))

			L = mask * I
			L = tf.reshape(L, (tf.shape(L)[0]//3,3,3))

		#_______________________________________________________________________________


		#old method - just consider principal axis length
		if method == 0:

			# get projections of axis length in each direction
			rotated = tf.abs(tf.matmul(U,axislen)) #was this pre 3/10
			# print("rotated", rotated)

			#need information on the cell index to be able to perform truncation 
			#	-> (cells further from vehicle will require larger distribution length thresholds)
			shell = cells//(self.fid_theta*(self.fid_phi - 1))
			# print("shell", shell)
			r_grid, _ = tf.unique(self.grid[:,0])
			# print("r_grid", r_grid)
			cell_width = tf.experimental.numpy.diff(r_grid)
			# print("cell_width", cell_width)
			# thresholds = (tf.gather(cell_width, shell)**2)/32
			# thresholds = (tf.gather(cell_width, shell)**2)/64 #was this
			thresholds = (tf.gather(cell_width, shell)**2) #NDT override


			#tile to so that each threshold is repeated 3 times (for each axis)
			thresholds = tf.reshape(tf.transpose(tf.reshape(tf.tile(thresholds[:,None], [3,1]), [3,-1])), [-1,3])[:,None]
			# print("thresholds", thresholds)

			greater_than_thresh = tf.math.greater(rotated, thresholds)
			# print(greater_than_thresh)
			ext_idx = tf.math.reduce_any(greater_than_thresh, axis = 1) #was this
			# ext_idx = tf.math.reduce_any(greater_than_thresh, axis = 2) #test
			compact = tf.where(tf.math.reduce_any(tf.reshape(ext_idx, (-1,1)), axis = 1) == False)
			compact =  tf.cast(compact, tf.int32)
			# print("compact", compact)
			data = tf.ones((tf.shape(compact)[0],3))
			I = tf.tile(tf.eye(3), (tf.shape(U)[0], 1))

			mask = tf.scatter_nd(indices = compact, updates = data, shape = tf.shape(I))

			L = mask * I
			L = tf.reshape(L, (tf.shape(L)[0]//3,3,3))

		return(U,L)

	def visualize_L(self, y0, U, L):
		""" for each voxel center, mu, this func draws untruncated axis via L 
			transformed into the frame of the distribution ellipsoids via U  """

		for i in range(tf.shape(y0)[0]):

			ends =  L[i] @ tf.transpose(U[i])
			# ends =  L[i] @ U[i] #WRONG!!


			arrow_len = 0.5
			arr1 = shapes.Arrow(y0[i].numpy(), (y0[i] + arrow_len * ends[0,:]).numpy(), c = 'red')
			self.disp.append(arr1)
			arr1 = shapes.Arrow(y0[i].numpy(), (y0[i] - arrow_len * ends[0,:]).numpy(), c = 'red')
			self.disp.append(arr1)

			arr2 = shapes.Arrow(y0[i].numpy(), (y0[i] + arrow_len * ends[1,:]).numpy(), c = 'green')
			self.disp.append(arr2)
			arr2 = shapes.Arrow(y0[i].numpy(), (y0[i] - arrow_len * ends[1,:]).numpy(), c = 'green')
			self.disp.append(arr2)
			
			arr3 = shapes.Arrow(y0[i].numpy(), (y0[i] + arrow_len * ends[2,:]).numpy(), c = 'blue')
			self.disp.append(arr3)
			arr3 = shapes.Arrow(y0[i].numpy(), (y0[i] - arrow_len * ends[2,:]).numpy(), c = 'blue')
			self.disp.append(arr3)

	def draw_DNN_soln(self, dnnsoln, itsoln, mu1):
		""" For each qualifying voxel, draw the solution vector estimated by the scan registation DNN 

			#dnnsoln = [n, 3] tensor with x, y, z translation estimates for each voxel
			itsoln = [n, 3] tensor, used to debug places where ICET and DNN solns differ greatly, want
							to make sure this works the same as our other perspective shift id technique
			mu1 = distribution centers from scan1 (only where sufficient correspondences occur)
			"""

		for i in range(tf.shape(dnnsoln)[0].numpy()):
			#normalize len of each arrow
			# arrowlen = 1/(np.sqrt(dnnsoln[i,0].numpy()**2 + dnnsoln[i,1].numpy()**2 + dnnsoln[i,2].numpy()**2))
			arrowlen = 1 #leave arrows proportional to residual distance
			# A = Arrow2D(startPoint = mu1[i].numpy(), endPoint = mu1[i].numpy() + arrowlen*dnnsoln[i,:].numpy(), c = 'purple')
			A = shapes.Arrow(mu1[i].numpy(), mu1[i].numpy() + arrowlen*dnnsoln[i,:].numpy(), c = 'purple')
			self.disp.append(A)

			#Draw ICET solns as well (for debug)
			# arrowlen = 1/(np.sqrt(itsoln[i,0].numpy()**2 + itsoln[i,1].numpy()**2 + itsoln[i,2].numpy()**2))
			B = shapes.Arrow(mu1[i].numpy(), mu1[i].numpy() + arrowlen*itsoln[i,:].numpy(), c = 'yellow')
			self.disp.append(B)


			# #draw big dot if dnnsoln and itsoln disagree
			# if (abs((dnnsoln[i] - itsoln[i]).numpy()) > 0.1).any():
			# 	# print(i)
			# 	dot = Points(np.array([[mu1[i,0].numpy(), mu1[i,1].numpy(), mu1[i,2].numpy()]]), c = "purple", r = 20)
			# 	self.disp.append(dot)


	def check_condition(self, HTWH):
		"""verifies that HTWH is invertable and if not, 
			reduces dimensions to make inversion possible

			L2 = identity matrix which keeps non-extended axis of solution
			lam = diagonal eigenvalue matrix
			U2 = rotation matrix to transform for L2 pruning 
			"""

		cutoff = 1e7 #1e4 #1e7 #TODO-> experiment with this to get a good value

		#do eigendecomposition
		eigenval, eigenvec = tf.linalg.eig(HTWH)
		eigenval = tf.math.real(eigenval)
		eigenvec = tf.math.real(eigenvec)

		# print("\n eigenvals \n", eigenval)
		# print("\n eigenvec \n", eigenvec)
		# print("\n eigenvec.T \n", tf.transpose(eigenvec))

		#sort eigenvals by size -default sorts small to big
		# small2big = tf.sort(eigenval)
		# print("\n sorted \n", small2big)

		#test if condition number is bigger than cutoff
		condition = eigenval[-1] / eigenval[0]
		# print("\n condition \n", tf.experimental.numpy.log10(abs(condition)))
		# print("\n condition \n", condition.numpy())


		everyaxis = tf.cast(tf.linspace(0,5,6), dtype=tf.int32)
		remainingaxis = everyaxis
		i = tf.Variable([0],dtype = tf.int32) #count var
		#loop until condition number is small enough to make matrix invertable
		while abs(condition) > cutoff:

			condition = eigenval[-1] / tf.gather(eigenval, i)
			# print("condition", tf.experimental.numpy.log10(abs(condition)).numpy())

			if abs(condition) > cutoff:
				i.assign_add(tf.Variable([1],dtype = tf.int32))
				remainingaxis = everyaxis[i.numpy()[0]:]

		#create identity matrix truncated to only have the remaining axis
		L2 = tf.gather(tf.eye(6), remainingaxis)

		# #alternate strategy- zero out instead of keeping axis truncated
		# while tf.shape(L2)[0] < 6:
		# 	L2 = tf.concat((tf.zeros([1,6]), L2), axis = 0)

		# print("\n L2 \n", L2)

		U2 = eigenvec
		# print("\n U2^T \n", tf.transpose(U2))

		#TODO: scale eigenvectors associated with rotational components of solution

		lam = tf.eye(6)*eigenval
		# print("\n lam \n", lam)

		return(L2, lam, U2)


	def draw_ell(self, mu, sigma, pc = 1, alpha = 1):
		"""draw distribution ellipses given mu and sigma tensors"""

		if pc == 1:
			# color = [0.8, 0.3, 0.3]
			color = '#a65852' #[0.8, 0.5, 0.5] #red
		if pc ==2:
			# color = [0.3, 0.3, 0.8]
			color = '#2c7c94' #[0.5, 0.5, 0.8] #blue

		for i in range(tf.shape(sigma)[0]):

			eig = np.linalg.eig(sigma[i,:,:].numpy())
			eigenval = eig[0] #correspond to lengths of axis
			eigenvec = eig[1]

			# assmues decreasing size
			a1 = eigenval[0]
			a2 = eigenval[1]
			a3 = eigenval[2]

			if mu[i,0] != 0 and mu[i,1] != 0:
				ell = Ell(pos=(mu[i,0], mu[i,1], mu[i,2]), axis1 = 4*np.sqrt(abs(a1)), 
					axis2 = 4*np.sqrt(abs(a2)), axis3 = 4*np.sqrt(abs(a3)), 
					angs = (np.array([-R2Euler(eigenvec)[0], -R2Euler(eigenvec)[1], -R2Euler(eigenvec)[2] ])), c=color, alpha=alpha, res=12)
				
				self.disp.append(ell)

	def draw_correspondences(self, mu1, mu2, corr):
		""" draw arrow between distributions between scans that:
			1- contain sufficient number of points 
			2- occupy the same voxel """

		# print("correspondences", corr)
		for i in corr:
			a = shapes.Arrow(mu2[i].numpy(), mu1[i].numpy(), c = "black", s = 0.005) #s = 0.01 #for thick lines
			self.disp.append(a)


	def get_corners_cluster(self, occupied_spikes, bounds):
		""" get 8 corners of region bounded by spike IDs and radial bounds """

		#spike IDs are the same as as cell IDs for the innermost shell of self.grid
		# 		so we can use that to get the theta and phi components
		corn = self.get_corners(occupied_spikes).numpy()
		# print("corn temp\n", np.shape(corn))

		#replace rad in grid with bounds
		corn[:,0,0] = bounds[:,0].numpy()
		corn[:,1,0] = bounds[:,0].numpy()
		corn[:,2,0] = bounds[:,1].numpy()
		corn[:,3,0] = bounds[:,1].numpy()
		corn[:,4,0] = bounds[:,0].numpy()
		corn[:,5,0] = bounds[:,0].numpy()
		corn[:,6,0] = bounds[:,1].numpy()
		corn[:,7,0] = bounds[:,1].numpy()

		return(corn)


	def get_corners(self, cells, tophat = 0):
		""" returns  spherical coordinates of coners of each input cell 
			cells = tensor containing cell indices """

		#to account for wrapping around at end of each ring
		per_shell = self.fid_theta*(self.fid_phi - 1) #number of cells per radial shell
		fix =  (self.fid_phi*self.fid_theta)*((((cells)%per_shell) + (self.fid_phi-1) )//per_shell)
		n = cells + cells//(self.fid_phi - 1)

		if tophat == 1:
			g = self.grid_tophat
		else:
			g = self.grid

		p1 = tf.gather(g, n)
		p2 = tf.gather(g, n+self.fid_phi - fix)
		p3 = tf.gather(g, n + self.fid_theta*self.fid_phi)
		p4 = tf.gather(g, n + self.fid_phi + (self.fid_theta*self.fid_phi) - fix)
		p5 = tf.gather(g, n + 1)
		p6 = tf.gather(g, n+self.fid_phi +1 - fix)
		p7 = tf.gather(g, n + (self.fid_theta*self.fid_phi) + 1)
		p8 = tf.gather(g, n + self.fid_phi + (self.fid_theta*self.fid_phi) +1 - fix)

		#NEW test 3/13 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# #for bottom ring in each shell, expand the radius of each cell
		# temp = p1[:,0] + tf.cast( (self.fid_phi-1 - cells%(self.fid_phi-1))//(self.fid_phi-1), tf.float32)
		# p1 = tf.concat((temp[:,None], p1[:,1:]), axis = 1)

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		out = tf.transpose(tf.Variable([p1, p2, p3, p4, p5, p6, p7, p8]), [1, 0, 2])

		return(out)


	def fit_gaussian(self, cloud, rag, npts):
		""" fits 3D gaussian distribution to each elelment of 
			rag, which cointains indices of points in cloud """

		coords = tf.gather(cloud, rag)

		xpos = tf.gather(cloud[:,0], rag)
		ypos = tf.gather(cloud[:,1], rag)
		zpos = tf.gather(cloud[:,2], rag)
		# print("mux",mu[:,0])
		# print("took", time.time()-st, "s to tf.gather")
		st = time.time()

		#if GPU is not available
		if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
			#old (works on CPU but SLOW)
			mu = tf.math.reduce_mean(coords, axis=1)
			# print("mu", tf.shape(mu))
			# print("mu[:,0][:,None]", tf.shape(mu[:,0][:,None]))
			# print("xpos", tf.shape(xpos))
			xx = tf.math.reduce_sum(tf.math.square(xpos - mu[:,0][:,None] ), axis = 1)/npts
			yy = tf.math.reduce_sum(tf.math.square(ypos - mu[:,1][:,None] ), axis = 1)/npts
			zz = tf.math.reduce_sum(tf.math.square(zpos - mu[:,2][:,None] ), axis = 1)/npts
			xy = tf.math.reduce_sum( (xpos - mu[:,0][:,None])*(ypos - mu[:,1][:,None]), axis = 1)/npts  #+
			xz = tf.math.reduce_sum( (xpos - mu[:,0][:,None])*(zpos - mu[:,2][:,None]), axis = 1)/npts #-
			yz = tf.math.reduce_sum( (ypos - mu[:,1][:,None])*(zpos - mu[:,2][:,None]), axis = 1)/npts #-
			sigma = tf.Variable([xx, xy, xz,
								 xy, yy, yz,
								 xz, yz, zz]) 
			sigma = tf.reshape(tf.transpose(sigma), (tf.shape(sigma)[1] ,3,3))
			return(mu, sigma)

		#if GPU is available
		else:
			#new method downsampling to first n points in each ragged tensor -- MUCH FASTER (but only works on GPU)
			mu = tf.math.reduce_mean(coords, axis = 1)[:,None]
			idx = tf.range(self.min_num_pts)
			# idx = tf.range(self.min_num_pts-1) #test
			xpos = tf.gather(xpos, idx, axis = 1)
			ypos = tf.gather(ypos, idx, axis = 1)
			zpos = tf.gather(zpos, idx, axis = 1)

			xx = tf.math.reduce_sum(tf.math.square(xpos - mu[:,:,0] ), axis = 1)/self.min_num_pts
			yy = tf.math.reduce_sum(tf.math.square(ypos - mu[:,:,1] ), axis = 1)/self.min_num_pts
			zz = tf.math.reduce_sum(tf.math.square(zpos - mu[:,:,2] ), axis = 1)/self.min_num_pts
			xy = tf.math.reduce_sum( (xpos - mu[:,:,0])*(ypos - mu[:,:,1]), axis = 1)/self.min_num_pts
			xz = tf.math.reduce_sum( (xpos - mu[:,:,0])*(zpos - mu[:,:,2]), axis = 1)/self.min_num_pts
			yz = tf.math.reduce_sum( (ypos - mu[:,:,1])*(zpos - mu[:,:,2]), axis = 1)/self.min_num_pts

			sigma = tf.Variable([xx, xy, xz,
								 xy, yy, yz,
								 xz, yz, zz]) 
			sigma = tf.reshape(tf.transpose(sigma), (tf.shape(sigma)[1] ,3,3))

			return(mu[:,0,:], sigma)


	def get_points_inside(self, cloud, cells):
		""" returns ragged tensor containing the indices of points in <cloud> inside each cell in <cells>"""
		st = time.time()
		# print("specified cells:", cells)

		thetamin = -np.pi
		thetamax = np.pi #-  2*np.pi/self.fid_theta
		phimin =  3*np.pi/8
		# phimin =  1*np.pi/8
		phimax = 7*np.pi/8 #why is this not the same as in <grid_spherical>????


		edges_phi = tf.linspace(phimin, phimax, self.fid_phi) #was this for regular cells
		# edges_phi, _ = tf.unique(self.grid[:,2])
		bins_phi = tfp.stats.find_bins(cloud[:,2], edges_phi)

		#works for regular voxels only--------------------------
		edges_r, _ = tf.unique(self.grid[:,0]) 
		bins_r = tfp.stats.find_bins(cloud[:,0], edges_r) 
		#-------------------------------------------------------

		#for extended radius brim voxels------------------------

		# # #TODO - need to modify the code for "get_occupied()"

		# # #temporarily half the radius measurement of every point with a phi value that puts it in the lower n "brim" bins to keep indexing working
		# temp_r = (cloud[:,0] - 3)*(1 - (cloud[:,2]//edges_phi[-3])/2) + 3
		# # print(temp_r[:,None])
		# # print(cloud[:,1:])
		# test = tf.concat((temp_r[:,None], cloud[:,1:]), axis = 1)
		# # print(test)
		# # print(edges_phi[-3])
		# # print((cloud[:,2]//edges_phi[-3]))
		# self.disp.append(Points(self.s2c(test).numpy(), c = 'green', r = 5 ))
		# edges_r, _ = tf.unique(self.grid[:,0])
		# bins_r = tfp.stats.find_bins(temp_r, edges_r) 
		#-------------------------------------------------------

		edges_theta = tf.linspace(thetamin, thetamax, self.fid_theta + 1)
		# edges_theta, _ = tf.unique(self.grid[:,1])
		bins_theta = tfp.stats.find_bins(cloud[:,1], edges_theta)
		# print("edges_theta", edges_theta)
		# print("bins_theta", bins_theta)		

		#cell index for every point in cloud
		cell_idx = tf.cast( bins_theta*(self.fid_phi-1) + bins_phi + bins_r*self.fid_theta*(self.fid_phi-1), tf.int32) #works for regular cells

		# print("cell index for each point", cell_idx)

		pts_in_c = tf.where(cell_idx == cells)
		# print("cell ID for each point", pts_in_c[:,0])

		#TODO: check for skipped indices here -> it may be producing a bug
		# print(tf.unique(pts_in_c[:,0]))
		
		numPtsPerCell = tf.math.bincount(tf.cast(pts_in_c[:,0], tf.int32))
		# print("numPtsPerCell", numPtsPerCell)
		#test
		# _, num_pts = tf.unique(pts_in_c[:,0])
		# print("num_pts", num_pts)

		pts_in_c = tf.RaggedTensor.from_value_rowids(pts_in_c[:,1], pts_in_c[:,0]) 
		# pts_in_c = tf.RaggedTensor.from_value_rowids(pts_in_c[:,1], pts_in_c[:,0], nrows = tf.shape(cells)[0].numpy())
		# print(pts_in_c.get_shape())
		# print(tf.shape(cells))
		# print("pts_in_c", pts_in_c)

		# print("index of points in specified cell", pts_in_c)

		# print("took", time.time()-st, "s to find pts in cells")
		return(pts_in_c, numPtsPerCell)


	def get_occupied(self, tophat = 0):
		""" returns idx of all voxels that occupy the line of sight closest to the observer """

		st = time.time()

		#attempt #2:------------------------------------------------------------------------------
		#bin points by spike
		thetamin = -np.pi
		thetamax = np.pi #-  2*np.pi/self.fid_theta
		phimin =  3*np.pi/8
		# phimin =  1*np.pi/8
		phimax = 7*np.pi/8 #why is this not the same as in <grid_spherical>????

		edges_phi = tf.linspace(phimin, phimax, self.fid_phi)
		edges_theta = tf.linspace(thetamin, thetamax, self.fid_theta + 1)

		cloud = self.cloud1_tensor_spherical
		if tophat == 1:
			temp_r = (cloud[:,0] - 3)*(1 - (cloud[:,2]//edges_phi[-3])/2) + 3
			cloud = tf.concat((temp_r[:,None], cloud[:,1:]), axis = 1)

		bins_theta = tfp.stats.find_bins(cloud[:,1], edges_theta)
		# print(bins_theta)
		bins_phi = tfp.stats.find_bins(cloud[:,2], edges_phi)
		# print(bins_phi)

		#combine bins_theta and bins_phi to get spike bins
		bins_spike = tf.cast(bins_theta*(self.fid_phi-1) + bins_phi, tf.int32)
		# print(tf.unique(bins_spike))
		# print("bins_spike", bins_spike)
		# self.draw_cell(tf.cast(bins_spike, tf.int32))

		#save which spike each point is in to ICET object for further analysis
		self.bins_spike = bins_spike

		#find min point in each occupied spike
		occupied_spikes, idxs = tf.unique(bins_spike)
		# print("occupied_spikes:", occupied_spikes)
		# print(idxs)

		temp =  tf.where(bins_spike == occupied_spikes[:,None]) #TODO- there has to be a better way to do this... 
		# print(temp)
		rag = tf.RaggedTensor.from_value_rowids(temp[:,1], temp[:,0])
		# print(rag)

		idx_by_rag = tf.gather(cloud[:,0], rag)
		# print(idx_by_rag)

		min_per_spike = tf.math.reduce_min(idx_by_rag, axis = 1)
		# print("min_per_spike:", min_per_spike)

		#get closest shell for each point in min_per_spike
		# print(tf.unique(self.grid[:,0]))
		radii, _ = tf.unique(self.grid[:,0])
		# print(radii)
		shell_idx = tf.math.reduce_sum(tf.cast(tf.greater(min_per_spike, radii[:,None] ), tf.int32), axis = 0) - 1
		# print(shell_idx)

		#find bin corresponding to the identified closeset points per cell
		occupied_cells = occupied_spikes + shell_idx*self.fid_theta*(self.fid_phi -1)
		# print("occupied_cells:", occupied_cells)

		# print("took", time.time() - st, "s to find occupied cells")
		return(occupied_cells)


	def draw_cell(self, corners, bad = False):
		""" draws cell provided by corners tensor"""

		# corners = self.get_corners(idx)
		# print(corners)

		if bad == False:
			for i in range(tf.shape(corners)[0]):
				p1, p2, p3, p4, p5, p6, p7, p8 = self.s2c(corners[i]).numpy()

				lineWidth = 1
				c1 = 'black'
				alpha_box = 0.3

				# arc1 = shapes.Arc(center = [0,0,0], point1 = p1, point2 = p2, c = 'red')	
				arc1 = shapes.Line(p1, p2, c = c1, lw = lineWidth, alpha = alpha_box) #debug		
				self.disp.append(arc1)
				# arc2 = shapes.Arc(center = [0,0,0], point1 = p3, point2 = p4, c = 'red')
				arc2 = shapes.Line(p3, p4, c = c1, lw = lineWidth, alpha = alpha_box) #debug
				self.disp.append(arc2)
				line1 = shapes.Line(p1, p3, c = c1, lw = lineWidth, alpha = alpha_box)
				self.disp.append(line1)
				line2 = shapes.Line(p2, p4, c = c1, lw = lineWidth, alpha = alpha_box) #problem here
				self.disp.append(line2)

				# arc3 = shapes.Arc(center = [0,0,0], point1 = p5, point2 = p6, c = 'red')		
				arc3 = shapes.Line(p5, p6, c = c1, lw = lineWidth, alpha = alpha_box) #debug
				self.disp.append(arc3)
				# arc4 = shapes.Arc(center = [0,0,0], point1 = p7, point2 = p8, c = 'red')
				arc4 = shapes.Line(p7, p8, c = c1, lw = lineWidth, alpha = alpha_box) #debug
				self.disp.append(arc4)
				line3 = shapes.Line(p5, p7, c = c1, lw = lineWidth, alpha = alpha_box)
				self.disp.append(line3)
				line4 = shapes.Line(p6, p8, c = c1, lw = lineWidth, alpha = alpha_box)
				self.disp.append(line4)

				self.disp.append(shapes.Line(p1,p5, c = c1, lw = lineWidth, alpha = alpha_box))
				self.disp.append(shapes.Line(p2,p6, c = c1, lw = lineWidth, alpha = alpha_box))
				self.disp.append(shapes.Line(p3,p7, c = c1, lw = lineWidth, alpha = alpha_box))
				self.disp.append(shapes.Line(p4,p8, c = c1, lw = lineWidth, alpha = alpha_box))

		if bad == True:
			#identified as containing moving objects
			for i in range(tf.shape(corners)[0]):
				p1, p2, p3, p4, p5, p6, p7, p8 = self.s2c(corners[i]).numpy()
				thicc = 3

				# arc1 = shapes.Arc(center = [0,0,0], point1 = p1, point2 = p2, c = 'yellow')	
				# arc1.lineWidth(thicc)
				arc1 = shapes.Line(p1, p2, c = 'yellow', lw = thicc) #debug
				self.disp.append(arc1)
				# arc2 = shapes.Arc(center = [0,0,0], point1 = p3, point2 = p4, c = 'yellow')
				# arc2.lineWidth(thicc)
				arc2 = shapes.Line(p3, p4, c = 'yellow', lw = thicc) #debug
				self.disp.append(arc2)
				line1 = shapes.Line(p1, p3, c = 'yellow', lw = thicc)
				self.disp.append(line1)
				line2 = shapes.Line(p2, p4, c = 'yellow', lw = thicc) #problem here
				self.disp.append(line2)

				# arc3 = shapes.Arc(center = [0,0,0], point1 = p5, point2 = p6, c = 'yellow')
				# arc3.lineWidth(thicc)
				arc3 = shapes.Line(p5, p6, c = 'yellow', lw = thicc) #debug			
				self.disp.append(arc3)
				# arc4 = shapes.Arc(center = [0,0,0], point1 = p7, point2 = p8, c = 'yellow')
				# arc4.lineWidth(thicc)
				arc4 = shapes.Line(p7, p8, c = 'yellow', lw = thicc) #debug
				self.disp.append(arc4)
				line3 = shapes.Line(p5, p7, c = 'yellow', lw = thicc)
				self.disp.append(line3)
				line4 = shapes.Line(p6, p8, c = 'yellow', lw = thicc)
				self.disp.append(line4)

				self.disp.append(shapes.Line(p1,p5,c = 'yellow', lw = thicc))
				self.disp.append(shapes.Line(p2,p6,c = 'yellow', lw = thicc))
				self.disp.append(shapes.Line(p3,p7,c = 'yellow', lw = thicc))
				self.disp.append(shapes.Line(p4,p8,c = 'yellow', lw = thicc))

		if bad == 2:
			#identified as perspective shift
			for i in range(tf.shape(corners)[0]):
				p1, p2, p3, p4, p5, p6, p7, p8 = self.s2c(corners[i]).numpy()
				thicc = 3

				# arc1 = shapes.Arc(center = [0,0,0], point1 = p1, point2 = p2, c = 'yellow')	
				# arc1.lineWidth(thicc)
				arc1 = shapes.Line(p1, p2, c = 'purple', lw = thicc) #debug
				self.disp.append(arc1)
				# arc2 = shapes.Arc(center = [0,0,0], point1 = p3, point2 = p4, c = 'yellow')
				# arc2.lineWidth(thicc)
				arc2 = shapes.Line(p3, p4, c = 'purple', lw = thicc) #debug
				self.disp.append(arc2)
				line1 = shapes.Line(p1, p3, c = 'purple', lw = thicc)
				self.disp.append(line1)
				line2 = shapes.Line(p2, p4, c = 'purple', lw = thicc) #problem here
				self.disp.append(line2)

				# arc3 = shapes.Arc(center = [0,0,0], point1 = p5, point2 = p6, c = 'yellow')
				# arc3.lineWidth(thicc)
				arc3 = shapes.Line(p5, p6, c = 'purple', lw = thicc) #debug			
				self.disp.append(arc3)
				# arc4 = shapes.Arc(center = [0,0,0], point1 = p7, point2 = p8, c = 'yellow')
				# arc4.lineWidth(thicc)
				arc4 = shapes.Line(p7, p8, c = 'purple', lw = thicc) #debug
				self.disp.append(arc4)
				line3 = shapes.Line(p5, p7, c = 'purple', lw = thicc)
				self.disp.append(line3)
				line4 = shapes.Line(p6, p8, c = 'purple', lw = thicc)
				self.disp.append(line4)

				self.disp.append(shapes.Line(p1,p5,c = 'purple', lw = thicc))
				self.disp.append(shapes.Line(p2,p6,c = 'purple', lw = thicc))
				self.disp.append(shapes.Line(p3,p7,c = 'purple', lw = thicc))
				self.disp.append(shapes.Line(p4,p8,c = 'purple', lw = thicc))


	def grid_spherical(self, draw = False):
		""" constructs grid in spherical coordinates """

		self.fid_r = self.fid  #waaayyy too many but keeping this for now
		self.fid_theta = self.fid  #number of subdivisions in horizontal directin
		self.fid_phi = self.fid_theta // 3

		thetamin = -np.pi 
		thetamax = np.pi - 2*np.pi/self.fid_theta #different from limits in main()
		phimin =  3*np.pi/8
		# phimin =  1*np.pi/8
		phimax = 7*np.pi/8

		a = tf.cast(tf.linspace(0,self.fid_r-1, self.fid_r)[:,None], tf.float32)
		b = tf.linspace(thetamin, thetamax, self.fid_theta)[:,None]
		c = tf.linspace(phimin, phimax, self.fid_phi)[:,None]

		ansb = tf.tile(tf.reshape(tf.tile(b, [1,self.fid_phi]), [-1,1] ), [(self.fid_r), 1])
		ansc = tf.tile(c, [self.fid_theta*self.fid_r, 1])
		#need to iteratively adjust spacing of radial positions to make cells roughly cubic

		nshell = self.fid_theta*(self.fid_phi) #number of grid cells per shell
		r_last = self.min_cell_distance #radis of line from observer to previous shell
		temp = np.ones([tf.shape(ansc)[0], 1])*self.min_cell_distance
		for i in range(1,self.fid_r):
			r_new = r_last*(1 + (np.arctan(2*np.pi/self.fid_theta))) #(cubic)
			# r_new = (r_last*(1 + (np.arctan(2*np.pi/self.fid_theta)))- 3)* 1.25 + 3 #(stretched)
			temp[(i*nshell):((i+1)*nshell+1),0] = r_new
			r_last = r_new
		ansa = tf.convert_to_tensor(temp, tf.float32)

		self.grid = tf.cast(tf.squeeze(tf.transpose(tf.Variable([ansa,ansb,ansc]))), tf.float32)

		if draw == True:
			gp = self.s2c(self.grid.numpy())
			# print(gp)
			p = Points(gp, c = [0.3,0.8,0.3], r = 5)
			self.disp.append(p)

	def c2s(self, pts):
		""" converts points from cartesian coordinates to spherical coordinates """
		r = tf.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2)
		phi = tf.math.acos(pts[:,2]/r)
		theta = tf.math.atan2(pts[:,1], pts[:,0])

		out = tf.transpose(tf.Variable([r, theta, phi]))
		return(out)

	def s2c(self, pts):
		"""converts spherical -> cartesian"""

		x = pts[:,0]*tf.math.sin(pts[:,2])*tf.math.cos(pts[:,1])
		y = pts[:,0]*tf.math.sin(pts[:,2])*tf.math.sin(pts[:,1]) 
		z = pts[:,0]*tf.math.cos(pts[:,2])

		out = tf.transpose(tf.Variable([x, y, z]))
		# out = tf.Variable([x, y, z])
		return(out)

	def draw_cloud(self, points, pc = 1):

		if pc == 1:
			color = '#a65852' #[0.8, 0.5, 0.5] #red
			self.PointsObj1 = Points(points, c = color, r = 2.5, alpha = 0.5).legend("Keyframe Scan") #r = 2.5
			self.disp.append(self.PointsObj1)
		if pc == 2:
			color = '#2c7c94' #[0.5, 0.5, 0.8] #blue
			self.PointsObj2 = Points(points, c = color, r = 2.5, alpha = 0.5).legend("New Scan") #r = 2.5
			self.disp.append(self.PointsObj2)
		if pc == 3:
			color = [0.5, 0.8, 0.5]
			c = Points(points, c = color, r = 2.5, alpha = 1.) #r = 2.5
			self.disp.append(c)
		if pc == 4:
			#HD MAP
			self.PointsObj1 = Points(points, c = 'black', r = 2.5, alpha = 0.1).legend("HD Map") #was 0.05 alpha for first HD Map GIF
			self.disp.append(self.PointsObj1)

		

	def draw_car(self):
		# (used for making presentation graphics)
		fname = "honda.stl"
		# car = Mesh(fname).c("gray").rotate(90, axis = (0,0,1)).addShadow(z=-1.85) #old vedo
		car = Mesh(fname).c("gray").rotate(90, axis = (0,0,1))
		car.pos(1.4,1,-1.72)
		# car.rotate(-45, axis = (0,0,1)) #for curve scene
		car.addShadow(plane = 'z', point = -1.85, c=(0.5, 0.5, 0.5))
		# car.orientation(vector(0,np.pi/2,0)) 
		self.disp.append(car)
		#draw sphere at location of sensor
		# self.disp.append(Points(np.array([[0,0,0]]), c = [0.9,0.9,0.5], r = 10))

		# fname = "C:/Users/Derm/lidar.stl"
		# velodyne = Mesh(fname).c("gray").rotate(90, axis = (1,0,0)).scale(0.001)
		# velodyne.rotate(180, axis = (0,1,0))
		# velodyne.pos(0,0.1,0.01)
		# velodyne.addShadow(plane = 'z', point = -0.08, c = (0.2, 0.2, 0.2))
		# self.disp.append(velodyne)

		# print(car.rot)
