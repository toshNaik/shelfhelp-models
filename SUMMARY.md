# Paper Summaries
### This markdown file contains the summaries/takeways for the papers I read.

## [SLAM for Visually Impaired People: A Survey](https://arxiv.org/pdf/2212.04745.pdf)
#### Some user survey points that can be used to justify our design choices:
- “61.9% of our sample prefer to use wearable assistive technology, 28.6% handheld assistive technology, and the remainder a white cane for navigation” - We are currently using a modified white cane but are also considering using a device that can be worn around the neck or some other apparatus that can be snapped onto a shopping cart.

- “the participants who have desire for handheld assistive technologies have severe or total blindness visual impairment.”
    
- Participants also described that finding their way in an unknown environment was one of the most challenging tasks. They find it hard to find objects or detect obstacles in front of them in an unfamiliar environment.
    
- “Over 71% of participants stated speech feedback as a preference to receive information about their surroundings” - We are currently using speech to navigate the user in their environment. There was another paper [Fathi et al. 2022] that described using three dimensional spatial audio as a feedback. Spatial audio for feedback is a potential area to explore.
    
- “The participants stated that an ideal assistive technology is affordable, lightweight, easy to use, and in the form of a smartwatch or smartphone. However, some of them preferred the assistive technology to be in combination with a cane.” - Preferences are subjective, designing a system that is modular and can snap onto a white cane, a grocery cart, or can be used as a wearable would take into consideration all users.

- The participants expressed the difficulty they face in finding their way around large indoor space. This is relevant in a supermarket setting. The respondents mentioned that this a particularly hard challenge due to lack of auditory and tactile feedback. They also pointed out the challenge in reaching their destination while going up or down staircases. One respondent specifically pointed out the need of conveying information about the number of stairs.


## [Augmented Reality for the Visually Impaired: Navigation Aid and Scene Semantics for Indoor Use Cases](https://ieeexplore.ieee.org/document/9937109/) 
#### The main contribution of this paper is that the authors have described a system that would help visually impaired people in exploring unfamiliar environments using three-dimensional audio feedback.
- The system first finds an optimal route to the destination using path planning. 
- Following this a Guiding Game Object (GGO) will travel along this route, always in vicinity of the user. The GGO emits a sound while traveling along the path so the user can follow it.
- The problem with this work was that the authors never really tested the entire guidance system. The tests only involved test subjects inferring the direction a sound was coming from.

## [Learning Topometric Semantic Maps from Occupancy Grids](https://arxiv.org/pdf/2001.03676.pdf)
#### The authors have described an approach to detect door hypotheses from an occupancy grid map. Furthermore, these are then used to categorize areas of the map as either a room, door or corridor. This would be relevant to our use case where we would want to get semantic information out of the occupancy grid map of a grocery store.
- A sliding window, with stride 8, is first used to split the input map into grids of size 64x64. The authors describe this to be the more approriate operation to perform over resizing as the latter results in loss of information. 
- Each image patch is then passed through a binary CNN classifier that gives out confidence score depending on whether that input has a doorway or not. Emphasis is placed on higher recall along with higher accuracy. This results in higher false detections than missed ones. The authors hypothesize that false detections can be later corrected.
- The patches that have higher doorway confidence are passed through a clustering algorithm based on proximity to avoid overlap.
- Finally, a modified U-Net network is used to segment the images. It outputs a binary mask demarcating the doorway regions. This is followed by some post-processing techinques using minimum bounding rectangles and maker based water shedding to finalize the doorways.
- To distinguish between rooms and corridors, the authors make the assumption that doors are the only links between different rooms and corridors. 
- The authors then propose a numerical approach, which involves finding the centroid of an "entity", and the distance of the centroid from the doors. Higher door count and higher variance in distance will result in the "entity" being classified as a corridor and lower door count and lower variance would result in it being classified as a room.

## [”Is it There or Not?” Why Augmented White Canes Do Not Need to Provide Detailed Feedback about Obstacles](https://dl.acm.org/doi/fullHtml/10.1145/3547522.3547685)
#### Blind and visually impaired people were provided with augmented white canes (AWC) that provided information about obstacles with 3 levels of granularity. 1. binary, 2. torso or above, 3. knee, waist or head level. The AWC was equipped with ultrasonic sensors capable of alerting to elevated obstacles at three different granularities - knee-, waist- or head-level.
- Participants preferred output at lowest level of granularity i.e. binary. since the purpose is to inform if there is an obstacle or not.
- One participant said vertical information is too situational to be useful in everyday life. One can just walk around the obstacle regardless of vertical positioning.
- Accuracy to detect vertical information decreases with increased granularity.
- Participants agreed that increased detection range would be beneficial in open spaces.
- Participants said they preferred a preview range that is just beyond the horizontal reach of the cane.
- Visually impaired people do not seek to avoid all obstacles since they intentionally seek to interact with some obstacles to use it as a landmark for navigation. It helps to create a cognitive map of the environment. Only vibrations cannot substitute the rich feedback provided by white canes (size, material, etc.) and very long range alerts can become a nuisance.

## [CNN-SLAM: Real-time dense monocular SLAM with learned depth prediction](https://arxiv.org/pdf/1704.03489.pdf)
#### This paper proposes a novel approach that uses depth maps predicted from CNN together with depth measurements obtained from direct monocular SLAM. Furthermore, semantic labels are fused with dense SLAM.
- The CNN based depth prediction is only applied on a few key frames. These key frames are determined by visual distinctness. 
- The pixel wise semantic labels are obtained by outputing a number of channels that match with the number of categories.

## [Spatial Perception by Object-Aware Visual Scene Representation](https://ieeexplore.ieee.org/document/9022544)
#### The paper proposes fusing semantic features extracted from objects with conventional gemoetric representations. The authors claim that this "compensates for the aberrations in geometric representations". As per the authors
- Geometric representations are limited when the scene lacks reliable descriptors. This would impair localization.
- A feature augmentation module is introduced into the visual SLAM system. It fuses geometric features with distribution of pixel-wise class probabilities. 

## [Learning Semantic Place Labels from Occupancy Grids using CNNs](https://april.eecs.umich.edu/pdfs/goeddel2016iros_a.pdf)
#### The authors propose a CNN network that classifies LIDAR sensor data to doorway, corridor and room. The authors claim that this would aid in localization and navigation. According to the authors:
- Occupancy grid maps are similar to grayscale images. So, they should be adept at identifying features in grid images that would aid in classification. 
- The method introduced here can work online. Live data produced by the robot may be fed to the system to get semantic labels.
- The occupancy grids are created by carving out free space instead of the traditional format in which the space is assumed free until structure is observed.