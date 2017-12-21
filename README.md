# Music Player Associated with Ambient Sound Detection
This is our final project for CSE 570. It is an Android app containing a music player associated with ambient sound detection.

## Main Idea
Our main idea is to make the smartphone periodically listen the ambient sound, identify the environment and respond to user’s music playing experience immediately, automatically, and intelligently. In order to achieve this goal, we plan to build an Android app with smart acoustic controlling function by using machine learning techniques.

## Workflow diagram
![alt text](https://github.com/MotoLee/Music-Player-Associated-with-Ambient-Sound-Detection/tree/master/report/workflow.png "workflow")

## Components in the app
![alt text](https://github.com/MotoLee/Music-Player-Associated-with-Ambient-Sound-Detection/tree/master/report/app_screenshot.png "screenshot")

## Four scenarios are detected
Four labels — “silence”, “single voice”, “crowd voice”, and “ambient noise” are set to represent the four scenarios that are able to describe user’s most common environment. The difference among these four scenarios is basically upon the sound type, yet not the environment types (e.g. indoor or outdoor). Following are the detailed definition of four labels: 
* Silence: In default most quiet places are classified as the silence scenario. 
* Single Voice: This label is for the scenario where only one or two persons are talking around the user. Usually this happens in some places like the user stays in the study room with friends or listens to a speech in the lecture hall.
* Crowd Voice: The label is for the scenario where multiple human voices are occurring continuously around the user. Some cases like a shuttle full with people is a typical example for this scenario.
* Ambient noise: This label is for the scenario where there is no human voice but a specific sound that almost never stops disturbing the user.

## Collect background sound data in Android devices
The Android media framework provides basic media services, including two major tasks in our app — collecting audio signals from devices’ audio source, usually from microphone, and playing various format media files as well. Here, our app use the AudioRecord APIs to collect background sound and use the MediaPlayer APIs to play music. Like a human being, the AudioRecord module is the ear of an app, which is used to hear surrounding voices from its microphone, and the MediaPlayer moulde is the mouth of an app, which sings a beautiful song.

Since we need to get the raw data of input audio signals for further analysis, we use the AudioRecord API, which manages the audio resources for Java applications to record audio from the audio input hardware of the platform. The collected audio data is encoded in PCM (Pulse-code modulation) format. From the PCM-encoded audio data, we can easily retrieve the amplitude and frequency of the sound. When initializing, we need to setup parameters of the AudioRecord module, including:
* Audio source: microphone
* Sample Rate: 16000 Hz
* Audio format: PCM 16 bits

## Analyze the time/frequency domain of collected sound
Once sufficient sound data collected, our app thereafter calculates the average value of sound volume in dB and implements fast-fourier transform (FFT) to obtain the frequency-domain data. The average sound volume is mostly employed as a threshold to sort the collected sound into silence or non-silence classes. The data in frequency-domain can indicate whether the sound comes from single or multiple source. We assume that the sound from single source may mostly distribute in a relatively narrow frequency band when comparing to the frequency distribution of multi-source sound. Note that, for FFT implementation, we use the free source code in Java. [3]

## Human voice detection using TensorFlow
This function is executed whenever the sound data is not regarded as silence scenario. At this stage, we will use the third-party deep-learning toolkit, TensorFlow, as our deterministic function to decide whether the sound source is human or not. TensorFlow is an open-source library based on deep neural networks. We leverage its tutorial of human voice commands recognition to develop our detecting function. [1] We assume there should be high-frequency words in most human daily talks, and usually such words are either single or two syllables like “Yes”, “No”, “Go”, “Up”, etc.. Therefore we use TensorFlow to train a model that can detect these words on the fly. If the model indicates that a sound data contains anyone of the word bank, this data would be regarded as either single voice and multiple voice; otherwise it would be ambient noise.

## Scoring mechanism
Both the time/frequency domain analysis and human voice detection functions collaboratively figure out what the sound type is for the sound collected in a constant time. Since the time frame of sound collection/recognition is actually very short (less than 1 second), we set up a scoring mechanism to determine the final prediction in a given period. To be more specific, once the collected sound data is marked with one scenario label, our adds one point to that label. The scenario detection and scoring last for a constant time (e.g. 30 seconds), and when the time is up, our app checks which label gets the highest score. Then our app adjusts the music volume if the scenario around the user changes. This mechanism ensures that the volume adjustment would not be triggered too frequently, and the misjudgement can be considerably reduced.

## Criteria of Scenario Identification
Below are the classification rules for these scenarios:
* If the average volume of input audio is lower than 40 dB, the scenario at the moment is regarded as a *silence* environment.
* If the input audio cannot get enough score in the human voice detection but the volume is large enough, it is the environment with *ambient noise*.
* If the TensorFlow model identifies that the input audio is full of human voice, and the volume never exceeds the 75dB and is closed to a positive skewed distribution, it is regarded as *single voice* environment.
* If sound is full of human voice while the volume is negative skewed or normalized distribution, it is a *crowd voice* environment.
Each time the identification process continuously classifies the collected sound for 1 minutes, and after that our app adjusts the music volume in accordance of the prediction. In terms of *silence* and *single voice* labels, the music would be turn down; while the *crowd voice* label would make it louder. The default label is ambience noise.

## Performance Evaluation
we conduct some experiment to evaluate the performance. We select the most common environments based on our daily life experience. Experiments conducted for every place contains 5 individual tests, and each test takes exactly one minute to acquire the accumulated score. Then, we sum up the total score of each scenario for five individual tests in a place and then calculate the ratio among four decisions, as shown in the following table.

Following are the results:

|                | SAC Lobby: Full of people, echo and noiseLibrary  | North Reading Area: With low noise | NCS Lobby: Without people but constant noise | Discussion Room in NCS: With few people |
|----------------|:-------------------------------------------------:|:----------------------------------:|:--------------------------------------------:|:---------------------------------------:|
| Silence        |                         0%                        |                 0%                 |                      0%                      |                  79.3%                  |
| Single Voice   |                        0.7%                       |                0.2%                |                      0%                      |                    0%                   |
| Crowd Voice    |                       60.4%                       |                6.6%                |                     6.2%                     |                    0%                   |
| Ambience Noise |                       38.9%                       |                93.2%               |                     93.8%                    |                  20.7%                  |

|                	| Starbucks: Almost at closing time 	| Campus Shuttle Bus: Radio and engine sound 	| Bed Room: Silence except hitter noise 	|                              	|
|----------------	|:---------------------------------:	|:------------------------------------------:	|:-------------------------------------:	|------------------------------	|
| Silence        	|                 0%                	|                     0%                     	|                 98.8%                 	|                              	|
| Single Voice   	|                3.6%               	|                     0%                     	|                   0%                  	|                              	|
| Crowd Voice    	|               37.4%               	|                    89.3%                   	|                   0%                  	|                              	|
| Ambience Noise 	|                59%                	|                    10.7%                   	|                  1.2%                 	|                              	|

In conclusion, most of the experiments are close to our expectation. However, the experiment result should be related to time/place. Even at the same place, when we conduct experiments in different time, the results might be very different.
