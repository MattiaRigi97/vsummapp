import cv2
import numpy as np
from sklearn.cluster import KMeans
from moviepy.editor import ImageSequenceClip
import imageio

# import pickle
# import matplotlib.pyplot as plt

# CLUSTERING
##################################################################################################################
# K-Means algorithm with frame
def kmeans_frame_clustering(frames, num_centroids, hist, sampling_interval, printing = False):
	summary_frames = []
	frame_indices = []
	kmeans = KMeans(n_clusters=num_centroids).fit(hist)
	if printing: 
		print ("\n \t\t Clustering done")
		print ("\t\t Generating summary frames..")
	# transforms into cluster-distance space (n_cluster dimensional)
	hist_transform = kmeans.transform(hist)
	for cluster in range(num_centroids):
		if printing:
			print("\t\t Original frame number: %d" % (np.argmin(hist_transform.T[cluster])*sampling_interval))
		frame_indices.append(np.argmin(hist_transform.T[cluster]))
	# frames generated in sequence from original video
	frame_indices = sorted(frame_indices)
	summary_frames = [frames[i] for i in frame_indices]
	if printing:
		print ("\t\t Generated summary\n")
	return frame_indices, summary_frames

# FEATURES
##################################################################################################################
# Function for generating the bgr hist with the selected frames
def generate_bgr_hist(frames, num_bins, printing = False):
	if printing:
		print ("\t\t Generating linear Histrograms using OpenCV")
	channels=['b','g','r']
	hist=[]
	for frame in frames:
		feature_value=[cv2.calcHist([frame],[i],None,[num_bins],[0,256]) for i,col in enumerate(channels)]
		hist.append(np.asarray(feature_value).flatten())
	hist=np.asarray(hist)
	if printing:
		print ("\t\t Done generating!")
		print ("\t\t Shape of histogram: " + str(hist.shape))
	return hist

def read_video_info(filename):
	video = imageio.get_reader(filename)
	n_frame_video = video.count_frames()
	metadata = video.get_meta_data()
	fps = metadata['fps']
	return fps, n_frame_video

# IMPORT
##################################################################################################################
def open_video(filename, downscale_quality_factor, sampling_interval, printing = False):
	if printing:
		print("\nOpening " + str(filename) + " video! \n")
	
	video = imageio.get_reader(filename)
	n_frame_video = video.count_frames()
	fps = int(video.get_meta_data()['fps'])

    # Read all the frames
	#idx_frames = list(range(0,n_frame_video-1,1))
	frames = [video.get_data(i) for i in range(0,n_frame_video)]
	# print(frames[0].shape)

	if downscale_quality_factor == "Low":
		downscale = 4
	if downscale_quality_factor == "Medium":
		downscale = 2
	if downscale_quality_factor == "High":
		downscale = 1

	frame_shape = frames[0].shape
	new_w = int(int(frame_shape[0]) / downscale)
	new_h = int(int(frame_shape[1])  / downscale)

	new_frame_list = []
	for frame in frames:
		new_frame = cv2.resize(frame, (new_h, new_w))
		new_frame_list.append(new_frame)
    
	# Frames subset
	# idx_frames_sel = idx_frames[::sampling_interval]
	frames_sel = new_frame_list[::sampling_interval]
	# print(frames_sel[0].shape)

	if printing:
		print ("\tLength of video %d" % n_frame_video)
		print ("\tConsidered frames %d" % len(frames_sel))
		print ("\n")
	
	return video, frames, frames_sel, n_frame_video, fps
	# return video, new_frame_list, frames_sel, n_frame_video, fps

# OUTPUT
##################################################################################################################
# Function for writing the video from a sequence of frames
def write_video_from_frame(output_path, video_name, _, summary_skim_frames, fps, printing=False):
	if printing:
		print("\n \t\t Writing the video summary..")
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	height, width, _ = summary_skim_frames[0].shape
	filename = output_path + "\\" + video_name + "_skimvideo.mp4"
	out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
	for frame in summary_skim_frames:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		out.write(frame)
	out.release()
	if printing:
		print("\t\t Video summary of " + str(video_name) + " saved \n")
		print("\t " + "..\\" + filename[100:])
	return None

def make_gif(video_name, output_path, summary_frames, n):
	
	clip = ImageSequenceClip(list(summary_frames), fps=2)
	# print(output_path)
	# print(video_name)
	clip.write_gif(output_path + '\\' + video_name + "_" + str(n) + "_" + "storyboard.gif", fps=2)
	return None

# OTHER
##################################################################################################################
def check_feasibility(n_frame_video, sampling_interval, num_centroids, printing=False):
	if printing:
		if (n_frame_video/sampling_interval) < num_centroids:
			print ("\t\t Samples too less to generate such a large summary")
			print ("\t\t Changing to maximum possible centroids")
			num_centroids = n_frame_video/sampling_interval
		else:
			print("\t\t Feasible")
	return None






