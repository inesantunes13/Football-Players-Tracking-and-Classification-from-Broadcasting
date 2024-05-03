# Person Classification and Counting from Broadcast Footage

##  Goal
The idea of this project was to come up with a viable solution to identify the players visible in 3 clips that were provided. Receiving broadcast footage clips as input, the output should be the number of players visible from each team and referees and the ball location in each 5th frame. 
So, for each input clip, two outputs should be provided:

- a json file that contains a row at each 5th frame of the video clip, in the following structure:

{“frame”:0, “home_team”:5, “away_team”: 10, “refs”: 0, “ball_loc”:(90, 19, 30, 31) }
{“frame”: 5, “home_team”:5, “away_team”: 8, “refs”: 1, “ball_loc”:(90, 19, 30, 31)}
{“frame”: 10, “home_team”:4, “away_team”: 9, “refs”: 1, “ball_loc”:(90, 19, 30, 31)}

ball_loc should be the x,y, width and height of the ball.

- the output video with detections and team colours, i.e., a video with the bounding boxes of detection classified according to their team for each of the clips given as input. The ball should also be annotated.


## Approach 
### Step 1: Pre-trained model yolov5
The initial step to approach this problem was to choose a YOLO model that could do an initial detectiong of the objects found.
THe one used was pre-trained yolov5 from ultralytics with CoCo dataset. 

![Alt text](../media/yolov5_initial_detection.png?raw=true)

The pre-trained model allowed for each player to be detected and classified as a person, and the ball to be classified as a sports ball. 
However, it was missing a further distinction among the person class, into:
- referee
- team_1
- team_2

In the bottom line, this can be seen as a clustering problem. When we look at a game in the television, we easily distinguish between both teams and the referee. What we are subconsciously doing is a clustering approach and, although we may use different indicators, the main aspect that stands out is their jersey colors. Other characteristics are also being processed, such as the overall positiong in the field, the moving tendency during the game and, for referees, some also have a flag. 

Having this said, step 2 focused on a clustering approach. 

### Step 2: K-Means CLustering for objects classified as "person"

#### Retrieve results files from YOLO
Since the most relevant feature is the jersey color, we want to use k-means based on just one feature: the color histogram of the players.
For this, we ideally need:
- the crops of the players: eighter by accessing the bounding boxes coordinates (xywh) or by directly access the crops yolo performs
- the overall detection information per object 

To have these intermediate outputs, `detect.py` script from yolov5 was changed into `detect_modified.py`, which can be found inside the `yolov5/`  folder.
This script not only extracts the intermediate results using a `--save-crop` and `--save-txt arguments`, but also shows a different .txt structure:
- class
- xyxy
- xywh
saving the information with a name structure that further allows the correlation between each .txt file and the correspondent crop. 

Both formats of bounding box (xyxy and xywh) were kept to facilitate the integration of the final detection to generate the video with the new classifications. 

#### Process data and apply K-Means
The script where this happens is 
After retrieving all the necessary information, the data was processed into  a single dataframe and filtered so that only the objects classified as person (cls==0) were clustered. 
The color histogram was calculated for each cropped image and the final feature vectors were fed to the k-means algorithm, defined to have 3 clusters (team_1, team_2 and referee).

Only ball detection was kept intact from the pre-trained YOLO detection. ALl the other objects were re-assigned a class.  

The functions for these steps are defined in the script `classify_with_kmeans.py` which is then called inside the main jupyter notebool (`main.py`).

### Step 3: Final Output Generation with new classes 
After all the objects were re-assigned to a new class, the final step was to integrate this new information into a final video. For this, `yolov5/class_into_video.py`, based on the logic used in YOLO for the automatic generation of the output video, receives the video we want to re-classify and a .pkl file with the relevant information for each detected object (bounding-box information, class).

Furthermore, at every 5 frames, it also counts the number of people detected per team and reports on the ball location in pixels. This file is saved in a "Jsons" folder, along with the "labels" folder (with the new .txt files) and the "crop" folder.

![Alt text](../media/final_folder_structure.png?raw=true)


### Final conclusions:
The overall experiment was successfull. Here's an example of the final output video and the correpondent .json file:

![Alt text](../media/final_output_video.png?raw=true)


![Alt text](../media/final_output_json.png?raw=true)

However:
- one of the clips had more innaccuracies, probalby due to shaded area in the image 
- the referee classification was also limited to its smaller number of appearances (in comparison with the other classes)
- other techniques may be further applied to improve the approach, such as thresholding techniques to each cropped image before the use of the k-means. 
