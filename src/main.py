#### PROBLEM TO SOLVE

# Packages
#########################################################################################
# KIVY and KIVYMD Libraries
from kivy.uix.anchorlayout import AnchorLayout
from kivymd.app import MDApp
from kivy.lang import Builder
from kivymd.uix.dialog import MDDialog
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.picker import MDThemePicker
from kivy.uix.videoplayer import VideoPlayer
from kivymd.uix.filemanager import MDFileManager
from kivy.uix.screenmanager import ScreenManager, Screen,NoTransition
from kivy.properties import ObjectProperty, StringProperty
from kivymd.uix.datatables import MDDataTable
from kivy.core.window import Window
from kivymd.toast import toast
from kivy.metrics import dp
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.button import MDFlatButton

# from kivymd.uix.list import OneLineIconListItem
# from kivymd.uix.menu import MDDropdownMenu
# from kivy.uix.textinput import TextInput
# from kivy.uix.gridlayout import GridLayout
# from kivy.uix.button import Button
# from kivy.uix.popup import Popup
# from kivy.uix.image import Image
# from kivy.uix.widget import Widget
# from kivymd.uix import menu
# from requests.api import get
# from torch._C import layout
# from torch._C import Storage

# General library
import re
import os
import random
import requests
import datetime
from pathlib import Path

# Torch Library
from torch import nn
from torchvision import models

# Import functions for inference
from inference import *

# Import general support functions
from tialib import *

# Import helpers function
from helpers import init_helper
from helpers import data_helper
from modules.model_zoo import get_model

# Library for Firebase
import pyrebase

# Define the Firebase Config Object
#########################################################################################
# It's necessary to create a project; these lines are in settings, 
# installation and configuration of SDK
config = {
	"apiKey": "AIzaSyAKMzbRMm24Mn16ess_QpBSqT-yb1cVOcU",
	"authDomain": "vsummapp.firebaseapp.com",
	"databaseURL": "https://vsummapp-default-rtdb.firebaseio.com",
	"projectId": "vsummapp",
	"storageBucket": "vsummapp.appspot.com",
	"messagingSenderId": "898496946513",
	"appId": "1:898496946513:web:ad8445f48a0b9135b27370",
	"measurementId": "G-XKSG695LEL"
}

# Define the storage object for add/remove files
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

# These are the command to load or download file from Firebase storage
#########################################################################################
# path_on_cloud = "videos/Fire_Domino.mp4"
# path_local = r"C:\Users\matti\github\vsummapp\gallery\Fire_Domino.mp4"

# Upload File
# storage.child(path_on_cloud).put(path_local)

# Download File
# storage.child(path_on_cloud).download(filename = "domino.mp4", 
#												path = path_local)

#########################################################################################
## CLASS ORDER
#########################################################################################
# Login Page
# Navigation Menu
# Home (Main Class)
# Video Player
# GIF Player
# Video Annotation
# Retrieve User's Preferences
# Screen Manager
# Main APP

# Class for the Login Page
#########################################################################################
# This class can be enanched with a complete autentichation module that can verify the user/pw credential
class ScreenLogin(Screen):
	
	username = None

	def onEnter(self):
		""" Define a function that read and pass the username to other class"""
		
		if len(self.ids.user_login.text) > 3:
			self.username = self.ids.user_login.text
			# print(self.ids.user_login.text)
			self.manager.current = "screenhome"
			# Pass the username to other class
			self.manager.screen_home.get_user(self.username)
			self.manager.screen_get_annotation.get_user(self.username)
			self.manager.screen_annotation.get_user(self.username)
		else:
			toast("Enter a valid User, at least 3 characters")		


# Class that define the NavigationDrawer Menù
#########################################################################################
class ContentNavigationDrawer(BoxLayout):
    
    screen_manager = ObjectProperty()
    nav_drawer = ObjectProperty()

    info_dialog = None
    contact_dialog = None

    def show_app_info_dialog(self):
        """PopUp for App Info"""

        if not self.info_dialog:
            self.info_dialog = MDDialog(
                title = "App Info",
                text = """Mobile Application for Video Summarization\n\nThis application allow to create a summary for a video, producing another video, shorter than the original, or a sequence of images in a GIF format. The algorithms implemented are DSnet for the first case and VSUMM for producing a sequence of images. \n\nAt the end, this app allow the user to annotate videos, specifing the important moments, for a potential enanching of the algorithms that can takes in objective the user's preferences.""",
                auto_dismiss = True
            )
        self.info_dialog.open()
    
    def show_contact_info_dialog(self):
        """PopUp for Contact Info"""

        if not self.contact_dialog:
            self.contact_dialog = MDDialog(
                title = "Contact",
                text = "For any doubt or consideration, please contact me on this addres m.rigiroli@campus.unimib.it or on my LinkedIn Profile Mattia Rigiroli.",
                auto_dismiss = True
            )
        self.contact_dialog.open()


# Main Class 
#########################################################################################
# This class contain the linkage to the KV File and all the principal functions
class Home(Screen):

	# Declare the KV file
	def build(self):
		# Read the KV File
		return Builder.load_file("vsummapp.kv")

	# Define the FileManager
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		Window.bind(on_keyboard=self.events)
		self.manager_open = False
		self.file_manager = MDFileManager(
			exit_manager=self.exit_manager,
			select_path=self.select_path,
			preview = False,
			# Allowed extensions
			ext = [".mp4", ".png", ".gif"],
		)

	# Initialize some variables
	path = None											# Selected path (video, summary or folder path)
	username = None										# Username passed to the first login page
	quality = None 										# Downscale factor to consider for the summary production
	duration = None										# Constraint duration to consider for the summary production
	gallery = r"C:\Users\matti\github\vsummapp\gallery" # Default gallery path
	summary = r"C:\Users\matti\github\vsummapp\summary" # Default summary path
	obj = None											# Support variable for change the default folder

	def get_user(self, username):
		"""Function to read the username"""
		
		self.username = username
	
	# Multimedia Functions
	#---------------------------------------------------------------------------------------
	def onScreenVideo(self, btn, fileName):
		'''Go to VideoPlayer'''
		
		lista_video_form = ["mp4"]
		if self.path != None:
			if self.path.split("/")[-1].split(".")[-1] in lista_video_form:
				self.manager.list_of_prev_screens.append(btn.parent.name)
				self.manager.current = 'screenvideo'
				self.manager.screen_video.test_on_enter(self.path)
			else:
				toast("Select a video, other formats are not allowed")
		else:
			toast("Select a video")

	def onScreenGif(self, btn, fileName):
		'''Go to GIFPlayer'''
		
		lista_video_form = ["gif"]
		if self.path != None:
			if self.path.split("/")[-1].split(".")[-1] in lista_video_form:
				self.manager.list_of_prev_screens.append(btn.parent.name)
				self.manager.current = 'screengif'
				self.manager.screen_gif.test_on_enter(self.path)
			else:
				toast("Select a GIF, other formats are not allowed")
		else:
			toast("Select a GIF")

	def onAnnotationVideo(self, btn, fileName):
		'''Go to Annotation Screen'''
		
		lista_video_form = ["mp4"]
		if self.path != None:
			if self.path.split("/")[-1].split(".")[-1] in lista_video_form:
				self.manager.list_of_prev_screens.append(btn.parent.name)
				self.manager.current = 'screenannotation'
				self.manager.screen_annotation.test_on_enter(self.path)
			else:
				toast("Select a video, other formats are not allowed")
		else:
			toast("Select a video")
	
	def onGetAnnotation(self, btn, fileName):
		'''Go to Get Annotation Screen'''
		
		self.manager.list_of_prev_screens.append(btn.parent.name)
		self.manager.current = 'screengetannotation'
		self.manager.screen_get_annotation.test_on_enter(self.path)

	# def show_image_popup(self):
	#	"""Function that show one image in a popup"""
	#	
	#	img_ext = ["png", "jpg", "jpeg"]
	#	if self.path.split("/")[-1].split(".")[-1] in img_ext:
	#		pop = Popup(title=self.path.split("/")[-1], content=Image(source=self.path),
	#				size_hint=(None, None), size=(600, 600))
	#		pop.open()		
	#	else:
	#		toast("Select an image, other formats are not allowed")
		
	# FileManager Functions
	#---------------------------------------------------------------------------------------
	def file_manager_gallery_open(self):
		'''Open the file system to change the gallery folder'''

		self.obj = "change_gallery"
		self.file_manager.show(r"C:\Users\matti") 
		self.manager_open = True

	def file_manager_summary_open(self):
		'''Open the file system to change the summary folder'''

		self.obj = "change_summary"
		self.file_manager.show(r"C:\Users\matti") 
		self.manager_open = True

	def gallery_manager_open(self):
		'''Open the gallery to choose video to summarize'''

		if self.gallery != None:
			self.obj = "open"
			self.file_manager.show(self.gallery) 
			self.manager_open = True

	def summary_manager_open(self):
		'''Open the summary folder where summary/storyboard genereted are saved'''

		if self.summary != None:
			self.obj = "open"
			self.file_manager.show(self.summary) 
			self.manager_open = True

	def select_path(self, path):
		'''It will be called when you click on the file name or the catalog selection button.
		:type path: str;
		:param path: path to the selected directory or file'''

		if self.obj == "change_gallery":
			self.gallery = path
			self.exit_manager()
			toast(path)

		if self.obj == "change_summary":
			self.summary = path
			self.exit_manager()
			toast(path)
		
		if self.obj == "open":
			self.exit_manager()
			self.path = path  	# Add the selected path to the variable path
			toast(path) 		# Show in a popup the file path selected
			# To show the video name, when selected, in the label_path box (Home & Summary Screen)
			if self.path != None:
				self.ids.label_path_home.text = "Selected: " +  str(self.path.split("\\")[-1])
				self.ids.label_path_summary.text = "Selected: " + str(self.path.split("\\")[-1])
		

	def exit_manager(self, *args):
		'''Called when the user reaches the root of the directory tree.'''
		
		self.manager_open = False
		self.file_manager.close()

	def events(self, instance, keyboard, keycode, text, modifiers):
		'''Called when buttons are pressed on the mobile device.'''
		
		if keyboard in (1001, 27):
			if self.manager_open:
				self.file_manager.back()
		return True

	def set_options(self, value):
		"""Function for read and pass the quality and duration options to summary functions"""
		
		if value == "High":
			self.quality = "High"
		if value == "Medium":
			self.quality = "Medium"
		if value == "Low":
			self.quality = "Low"
		if value == "20%":
			self.duration = 0.20
		if value == "15%":
			self.duration = 0.15
		if value == "10%":
			self.duration = 0.10
		
		if self.quality != None:
			# Show the quality selected
			self.ids.my_quality_label.text = f'{self.quality} Selected'
		if self.duration != None:
			# Show the duration contraint selected
			self.ids.my_duration_label.text = f'{str(int(self.duration*100))}% Selected'

	def choose_random_video(self):
		"""Function to randomly choose a default video that the user has not yet annotated"""

		####  Add the possibility to specify a max duration of the video to annotate

		fb_annotation_url = "https://vsummapp-default-rtdb.firebaseio.com/Annotations/"
		fb_video_url = "https://vsummapp-default-rtdb.firebaseio.com/Default_Videos/"

		user = self.username
		# self.duration_limit = int(int(self.ids.random_duration_slider.value) * 60)
		# print(self.duration_limit)
		
		# Keep all the annotation url
		res = requests.get(url = fb_annotation_url + ".json")
		res_url = list(res.json()) 	# This list contain all the annotations for all users
		
		#### Retrieve only the user's annotations

		# Select only the url that contain the username
		annotated_video = []
		for i in range(0,len(res_url)):
			if user in res_url[i]:
				annotated_video.append(res_url[i].split("-")[-1]) # Extract video names

		# Read the list of all videos in default_videos folder on firebase storage
		res = requests.get(url = fb_video_url + ".json")
		list_default_video = list(res.json()) 
		# res_json = res.json()
		# print(res.json())
		# print(list_default_video)
		
		var = True
		while var:
			# Until a video is found
			selected_video = random.choice(list_default_video)
			# print(res_json[selected_video]["Time"])
			# print(self.duration_limit)
			if selected_video not in annotated_video:
				# if int(res_json[selected_video]["Time"]) < int(self.duration_limit):
				# print(selected_video)
				var = False
				#### Solve the problem when all video are annotated.. there is an infinite cycle...
				storage.child("default_videos/" + selected_video + ".mp4").download(filename = self.gallery + "\\" + selected_video + ".mp4", path = self.gallery + "\\" + selected_video + ".mp4")
				toast("Downloading video..")
				self.path = self.gallery + "\\" + selected_video + ".mp4"
				toast("Open the filemanager and select the downloaded video: " + str(selected_video))
				
		
	# Summary Generation Functions
	#---------------------------------------------------------------------------------------
	def make_summary(self):
		""" Function that apply DSNet Algorithm and create a video summary"""
		
		# Set the quality and duration variables
		if self.path != None:
			if self.quality != None and self.duration != None:
				perc_proportion = self.duration
				quality = self.quality
			else:
				perc_proportion = 0.15
				quality = "Low"
				
			# print("Quality selected: " + str(quality))
			# print("% of duration selected: " + str(perc_proportion*100) + "%")

			# Initializate the model
			args = init_helper.get_arguments()
			args.model = "anchor-based"
			model_name = "ab_mobilenet"
			seg_algo = "kts"
			args.splits = ['../splits/tvsum.yml', '../splits/summe.yml']
			
			# Video path to summarize
			filename = self.path
			video_name = filename.split("\\")[-1].split(".")[0]
			output_video_path = args.output_path

			toast("Making summary..")

			if self.path.split("/")[-1].split(".")[-1] == "mp4":
				# Load the model
				model = get_model(args.model, **vars(args))
				model = model.eval().to(args.device)
				for split_path in args.splits:
					split_path = Path(split_path)
					splits = data_helper.load_yaml(split_path)
					# For each split (train/test) in dataset.yml file (x5)
					for split_idx, _ in enumerate(splits):
						# Load the model from the checkpoint folder
						ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path, split_idx)
						state_dict = torch.load(str(ckpt_path), map_location=lambda storage, loc: storage)
						model.load_state_dict(state_dict)

				# Load video
				_, frames, frames_sel, n_frame_video, fps = open_video(filename, quality, sampling_interval=15)

				# Initialize the feature extractor model
				feat_extr = models.mobilenet_v2(pretrained=True)
				feat_extr.eval()
				new_classifier = nn.Sequential(*list(feat_extr.classifier.children())[:-2])
				feat_extr.classifier = new_classifier
				shape = feat_extr(torch.randn(1,3,224,224)).shape[1]
					
				# Change the device to GPU if available
				device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
				feat_extr = feat_extr.to(device)

				# Run inference function
				pred_summ = inference(model, feat_extr, shape, filename, frames_sel, n_frame_video, seg_algo, args.nms_thresh, args.device, proportion=perc_proportion)
				# Trasform and save in .mp4 file 
				pred_summ = np.array(pred_summ) # True False Mask
				frames = np.array(frames)
				final_summary = frames[pred_summ,:]
				# print("Final summary frames: " + str(final_summary.shape))

				# Write the .mp4 summary file
				write_video_from_frame(output_video_path, video_name, model_name, final_summary, fps)
				# Go to Gallery Screen
				toast("Summary generated")
				self.path = None
			else:
				toast("Select a video")
		else:
			toast("First select a video")

	def make_storyboard(self):
		"""Function that apply VSUMM keyframe extraction for producing static storyboard"""

		# Read the number of keyframe to utilize for storyboard production
		self.k = int(self.ids.select_k.value)
		# print(self.k)

		# Set the quality variable
		if self.quality != None:
			quality = self.quality
		else:
			quality = "Low"

		if self.k != None:
			if self.path != None:
				if self.path.split("/")[-1].split(".")[-1] == "mp4":
					# Open the selected video
					args = init_helper.get_arguments()
					filename = self.path
					output_path = args.output_path
					video_name = filename.split("\\")[-1].split(".")[0]
					
					# Storyboard options
					printing = False
					num_bins = 16
					sampling_interval = 15
					
					# Load the video
					_, _, frames, n_frame_video, _ = open_video(filename, quality, sampling_interval, printing)
					
					# Check the feasibility
					check_feasibility(n_frame_video, sampling_interval, self.k, printing)

					# GENERATE THE BGR HIST
					hist = generate_bgr_hist(frames, num_bins, printing)

					# K-MEANS CLUSTERING
					_, summary_frames = kmeans_frame_clustering(frames, num_centroids = self.k, hist = hist, sampling_interval=25)
								
					# Producing the GIF
					# plot_keyframe(video_name, output_path, summary_frames, self.k)
					make_gif(video_name, output_path, summary_frames, self.k)

					# Go to Gallery Screen
					toast("Storyboard generated")
		
				else:
					toast("Select a video")

			else:
				toast("First select a video")
		
		else:
			toast("Select a K for the storyboard production")
	

# Class for VideoPlayer
#########################################################################################
class ScreenVideo(Screen):

	# Initialize the VideoPlayer
	def test_on_enter(self, vidname):
		# print(vidname)
		# self.add_widget(Button(text="Back"))
		if vidname != None:
			self.vid = VideoPlayer(source=vidname, state='play',
									options={'allow_stretch':False,
											'eos': 'loop'}, 
									size_hint = (.8, .8), 
									pos_hint =  {"center_y": .5, "center_x": .5})
			self.add_widget(self.vid)
		else:
			self.manager.current = "screenhome"

	# Back Home
	def onBackBtn(self):
		self.vid.state = 'stop'
		self.remove_widget(self.vid)
		self.manager.current = "screenhome"		


# Class for GIF Player
#########################################################################################
class ScreenGif(Screen):

	# Initialize the Gif Screen
	def test_on_enter(self, vidname):
		if vidname != None:
			self.ids.gif.source = vidname
		else:
			self.manager.current = "screenhome"

	# Back Home
	def onBackBtn(self):
		self.manager.current = "screenhome"	


# Class for Annotate Video
#########################################################################################
class ScreenAnnotation(Screen):
	
	# Define some variables
	filename = None
	n_frame = None
	fps = None
	start = None
	end = None
	segment = None
	upload_video_dialog = None
		
	# Initialize the VideoPlayer
	def test_on_enter(self, vidname):
		#self.add_widget(Button(text="Back"))
		if vidname != None:
			self.vid = VideoPlayer(source=vidname, state='play',
									options={'allow_stretch':False,
											'eos': 'loop'}, 
									size_hint = (.5, .5), 
									pos_hint =  {"center_y": .5, "center_x": .3})
			# print(vidname)
			self.filename = vidname
			# Read the info about the video
			fps, n_frame = read_video_info(filename=vidname)
			self.fps = fps
			self.n_frame = n_frame
			self.add_widget(self.vid)
		else:
			self.manager.current = "screenannotation"

	def get_user(self, username):
		"""Function to read the username"""

		self.username = username

	# Back Home
	def onBackBtn(self):
		"""Come back and reset the annotation screen"""

		self.vid.state = 'stop'
		self.remove_widget(self.vid)
		self.manager.current = "screenhome"
		self.ids.annotation.text = " "
		self.ids.my_progress_bar.value = 0
		self.ids.my_label.text = '0% Progress'
		self.ids.label_start.text = str(0)
		self.ids.label_end.text = str(0)

	def get_time_position_start(self):
		"""Get the start position selected"""

		# print(round(self.vid.position),2)
		self.start = self.vid.position 
		self.ids.label_start.text = str(int(round(self.start,0)))

	def get_time_position_end(self):
		"""Get the end position selected"""

		# print(round(self.vid.position),2)
		self.end = self.vid.position 
		self.ids.label_end.text = str(int(round(self.end,0)))

	def add_segment(self):
		"""Add the segment selected (if valid) in the annotation box"""

		if self.start != None and self.end != None:
			if self.end > self.start:
				self.segment = self.ids.annotation.text
				self.segment = self.ids.annotation.text + str(round(self.start,2)) + ", " + str(round(self.end,2)) + "\n"
				self.ids.annotation.text = self.segment
			else:
				toast("Select a valid segment")

	def update_selection(self):
		"""Function that update the progress bar for display the %/total of frames allowed by duration constraint"""
		
		# Video info
		fps = self.fps
		n_frame = self.n_frame
		duration = 0.15 #### Allow to select the duration constraint for annotation?

		# Read the annotation text field (es. 10,12 \n 14,15 ..)
		annotation = self.ids.annotation.text
		temp = re.findall(r'\d*\.\d*', annotation)
		# print(temp)
		
		# Create a list of number that are the gt segments
		annotation = list(map(float, temp))
		# print(annotation)

		# Count the number of frames selected
		sel_frames = 0
		max_frame = n_frame * duration
		for i in range(0,len(annotation)):
			if i % 2 == 0:
				start = annotation[i]
				end = annotation[i+1]
				sel_frames += (end - start)*fps

		# Control constraint of duration % 
		if sel_frames > max_frame:
			toast("Too many frames selected")

		# Update the progress bar
		self.ids.my_progress_bar.value = int((sel_frames/max_frame)*100)
		# Update the label
		self.ids.my_label.ths = int((sel_frames/max_frame)*100)
		self.ids.my_label.text = f'{int((sel_frames/max_frame)*100)}% Progress'
	
	# Create functions to interact with DB
	#---------------------------------------------------------------------------------------------

	# Realtime Firebase Database
	firebase_annotation_url = "https://vsummapp-default-rtdb.firebaseio.com/Annotations/"
	firebase_video_url = "https://vsummapp-default-rtdb.firebaseio.com/Videos/"

	def create_patch(self, *args):
		"""Function to upload the annotation and if the video is not uploaded, at the end the video is uploaded"""

		# Control the thresold of the annotation constraint 
		if self.ids.my_label.ths > 100:
			toast("Too many frames selected")
		if self.ids.my_label.ths < 20:
			toast("Too few frames selected")
		else:	
			# Annotation
			user_name = self.username
			# Create the annotation key --> username-videoname
			key = user_name + "-" + str(self.filename.split("/")[-1].split(".")[0])
			now = datetime.datetime.now()
			date = datetime.datetime.strftime(now, '%m/%d/%Y')
			time = datetime.datetime.strftime(now, '%H:%M')
			json_data = {key: 
						{"Date": str(date), 
						"Time": str(time), 
						"Video_Name" : str(self.filename.split("/")[-1].split(".")[0]),
						"Annotation": self.ids.annotation.text}}
			# Upload the annotation
			res = requests.patch(url = self.firebase_annotation_url + ".json", json=json_data)
			toast("Annotation saved")
			# print(res)

			# Upload video control
			res = requests.get(url = self.firebase_video_url + ".json")
			# print(res.json())
			if res.json() != None:
				video_names = list(res.json())
			else:
				video_names = []
			
			# If the video is absent, upload the video into storage
			if str(self.filename.split("/")[-1].split(".")[0]) not in video_names:
				
				#### PopUp for Confirm the video upload
				"""
				if not self.upload_video_dialog:
					self.upload_video_dialog = MDDialog(
						title = "Upload Confirmation",
						buttons = [MDFlatButton(text = "Cancel", on_release=self.close_dialog()),
									MDFlatButton(text = "Accept"),],
						text = "Do you want to upload the video?",
					)
				self.upload_video_dialog.open()"""

				path_on_cloud = "videos/"+ str(self.filename.split("/")[-1].split(".")[0]) + ".mp4"
				storage.child(path_on_cloud).put(self.filename)

				# Add the video in the realtime database /Videos
				fps = self.fps
				n_frame = self.n_frame
				duration = int(n_frame / fps)
				json_data = {str(self.filename.split("/")[-1].split(".")[0]): {"Duration": duration}}    
				res = requests.patch(url = self.firebase_video_url + ".json", json = json_data) 

			# Reset the annotations screen
			self.ids.annotation.text = " "
			self.ids.my_progress_bar.value = 0
			self.ids.my_label.text = '0% Progress' 
			self.ids.label_start.text = str(0)
			self.ids.label_end.text = str(0)

	def delete_video_from_gallery(self):
		"""Function to delete video from gallery, expecially thinked for the random default video that 
		are downloaded only for annotatoin purpose"""

		self.vid.state = 'stop'
		self.remove_widget(self.vid)
		self.manager.current = "screenhome"
		# Reset the annotations screen
		self.ids.annotation.text = " "
		self.ids.my_progress_bar.value = 0
		self.ids.my_label.text = '0% Progress'
		self.ids.label_start.text = str(0)
		self.ids.label_end.text = str(0)
		# print(self.filename)
		os.remove(self.filename)

# Class that show the user's annotations in a separate screen within a table
class GetAnnotation(Screen):

	def get_user(self, username):
		"""Function to read the username"""

		self.username = username

	# Define some variables
	annotation_dialog = None
	one_annotation = None

	#### Optimize the search for user annotation by username
	def test_on_enter(self, vidname):

		"""Inizialization of GetAnnotation"""

		user = self.username
		fb_annotation_url = "https://vsummapp-default-rtdb.firebaseio.com/Annotations/"
		
		# Keep all the url
		res = requests.get(url = fb_annotation_url + ".json")
		res_url = list(res.json()) 	# This list contain all the annotations for all users

		# Select only the url that contain the username
		user_url = []
		for i in range(0,len(res_url)):
			if user in res_url[i]:
				user_url.append(fb_annotation_url + "/" + res_url[i] + '.json') 
		
		date = []
		time = []
		video_name = []
		annotation =  []

        # For each url related to an annotation
		for url in user_url:
			res = requests.get(url = url)
			#print(res.json())
			video_name.append(res.json()["Video_Name"])
			date.append(res.json()["Date"])
			time.append(res.json()["Time"])
			annotation.append(res.json()["Annotation"])

		self.annotation = annotation

        # Create the datatable with the founded annotations	
		#### Solve problem of visualization when there is only one items
		self.layout = AnchorLayout()
		self.data_tables = MDDataTable(
			check = True,
			size_hint=(0.9, 0.6),
			use_pagination=True,
			column_data=[
                ("Video ID", dp(30)),
                ("Video Name", dp(50)),
                ("Date", dp(50)),
				("Time", dp(20)),
            ],
			row_data=[
                (f"{i + 1}", video_name[i], str(date[i]), str(time[i])) for i in range(0,len(annotation))
            ],
        )
		self.layout.add_widget(self.data_tables)
		self.add_widget(self.layout)

	def removeSelectedRows(self, *args):
		"""Function for remove from table (and from the DB) the annotation row checked"""
	
		fb_annotation_url = "https://vsummapp-default-rtdb.firebaseio.com/Annotations/"

		# Read the checked rows
		rows2remove_raw = self.data_tables.get_row_checks()
		# print(rows2remove_raw)
		
		user_name = self.username
		key = user_name + "-" + str(rows2remove_raw[0][1])
		delete_url = fb_annotation_url + key + ".json"
		# print(key)

		res = requests.delete(url = delete_url)
		if res.ok:
			toast("Data Eliminated")
		else:
			raise("The key is absent")
		# print(res)

		# Keep all the url
		res = requests.get(url = fb_annotation_url + ".json")
		res_url = list(res.json()) 	# This list contain all the annotations for all users

		# Select only the url that contain the username
		user_url = []
		for i in range(0,len(res_url)):
			if user_name in res_url[i]:
				user_url.append(fb_annotation_url + "/" + res_url[i] + '.json') 
		
		date = []
		time = []
		video_name = []
		annotation =  []

        # For each url related to an annotation
		for url in user_url:
			res = requests.get(url = url)
			# print(res.json())
			video_name.append(res.json()["Video_Name"])
			date.append(res.json()["Date"])
			time.append(res.json()["Time"])
			annotation.append(res.json()["Annotation"])

        # Update the table
		self.layout = AnchorLayout()
		self.data_tables = MDDataTable(
            size_hint=(0.9, 0.6),
            use_pagination=True,
            check=True,
			column_data=[
                ("Video ID", dp(30)),
                ("Video Name", dp(50)),
                ("Date", dp(50)),
				("Time", dp(20)),
            ],
			row_data=[
                (f"{i + 1}", video_name[i], str(date[i]), str(time[i])) for i in range(0,len(annotation))
            ],
        )
		self.layout.add_widget(self.data_tables)
		self.add_widget(self.layout)

		self.annotation = annotation

	def show_annotation_dialog(self, *args):
		"""PopUp for show the annotations of one video"""

		get_annotation_raw = self.data_tables.get_row_checks()
		print(get_annotation_raw)
		id = int(get_annotation_raw[0][0]) - 1
		
		#print("ID " + str(get_annotation_raw[0][0]))
		#print(self.annotation[id])
		
		if len(get_annotation_raw) == 1:
			app_info = str(self.annotation[id])
			if not self.annotation_dialog:
				self.annotation_dialog = MDDialog(
					title = "Annotation",
					text = app_info,
					auto_dismiss = True
				)
			self.annotation_dialog.open()
			get_annotation_raw = None
		else:
			toast("Select only one row")
		# Reset the annotation_dialog 
		self.annotation_dialog = None

	# Back Home
	def onBackBtn(self):
		self.manager.current = "screenhome"

# Screen Manager Class
#########################################################################################
class Manager(ScreenManager):
	
	transition = NoTransition()
	screen_login = ObjectProperty(None)
	screen_home = ObjectProperty(None)
	screen_video = ObjectProperty(None)
	screen_annotation = ObjectProperty(None)
	screen_get_annotation = ObjectProperty(None)

	def __init__(self, *args, **kwargs):
		super(Manager, self).__init__(*args, **kwargs)
		# list to keep track of screens we were in
		self.list_of_prev_screens = []


# Main APP Class
#########################################################################################
class VSummApp(MDApp):

	state = StringProperty("stop")

	def show_theme_picker(self):
		"""Function that allow to switch the theme"""
		theme_dialog = MDThemePicker()
		theme_dialog.open()

	def build(self):	
		"""Build function that declare the principal design settings"""
		self.title = "Vid Summ App"
		self.theme_cls.theme_style = "Dark"
		self.theme_cls.primary_palette = "Gray"
		self.theme_cls.accent_palette = "Red"
		self.theme_cls.primary_hue = "300"
		return Manager()

	# User PopUp
	#---------------------------------------------------------------------------------------------

	def create_snackbar(self, text):

		snackbar = Snackbar(
			text = text,
			snackbar_x="10dp",
			snackbar_y="10dp",
		)
		snackbar.size_hint_x = (
			Window.width - (snackbar.snackbar_x * 2)
		) / Window.width
		snackbar.buttons = [
			MDFlatButton(
				text="Close",
				text_color=(1, 1, 1, 1),
				on_release=snackbar.dismiss,
			),
		]
		snackbar.open()

	duration_dialog = None
	quality_dialog = None
	k_dialog = None
	randomvideo = None
	annotation_page = None

	def show_app_quality_dialog(self):
		"""PopUp for Quality Info"""

		if not self.quality_dialog:
			self.quality_dialog = MDDialog(
			title = "Quality Settings",
                text = """Specify the quality that you want to consider for the summary production. High correspond to original, medium halves the quality and low applies a reduction factor of 4.""",
                auto_dismiss = True
            )
		self.quality_dialog.open()

	def show_app_duration_dialog(self):
		"""PopUp for Duration Info"""

		if not self.duration_dialog:
			self.duration_dialog = MDDialog(
			title = "Duration Settings",
                text = """Specify the duration constraint that must be respected by the alghorithm for the video skimming production. This parameter is not utilized for storyboard because we consder the K value.""",
                auto_dismiss = True
            )
		self.duration_dialog.open()

	def show_app_k_dialog(self):
		"""PopUp for K Info"""

		if not self.k_dialog:
			self.k_dialog = MDDialog(
			title = "K Settings",
                text = """Select the K number, that represents the number of images that must be considered for the storyboard production.""",
                auto_dismiss = True
            )
		self.k_dialog.open()

	def show_app_randomvideo_dialog(self):
		"""PopUp for .... Info"""

		if not self.randomvideo:
			self.randomvideo = MDDialog(
			title = "Download a random video",
                text = """If you want to annotate a video but have no idea which one, this will randomly choose a default video. The video will be downloaded on your device but after the annotation you can delete it!""",
                auto_dismiss = True
            )
		self.randomvideo.open()

	def show_app_annotation_dialog(self):
		"""PopUp for .... Info"""

		if not self.annotation_page:
			self.annotation_page = MDDialog(
			title = "Annotation Page",
                text = """This page allow to annotate the selected video. With start and end buttons, you can select a segment of the video that you consider relevant. Than, you can add the segment, with the button, that will appear in the annotation box. You can control the coverage of your annotation, that it must not be too short or long, with the control button. Finally, you have to press the save button to save the annotation, and the video if it's your, on the database.""",
                auto_dismiss = True
            )
		self.annotation_page.open()


if __name__ == "__main__":
	VSummApp().run()