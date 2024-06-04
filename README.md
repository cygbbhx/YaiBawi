# YaiBawi (ShellTrack)
### :1st_place_medal: YAICON 4rd 1st Prize
</br>
<h2 align="center"> :nesting_dolls: ShellTrack: Multi-Object Tracking in <br/>IDENTICAL Appearance and FAST Motion </h2>
</br>
<p align="center"><img src="https://github.com/cygbbhx/YaiBawi/blob/main/videos/demo.gif" width="50%" height="50%"></p>
</br>

## Updates

- [ ] Release our custom dataset.
- [ ] Release our custom detector weights.
- [ ] Finalize scripts for each experiment setup.
- [ ] Update additional experimental results.
- [x] 06/04/2024: *Initial version of code released.*

---
## Introduction
In Object Tracking, "ID Switch" refers to a problem where the identities of two or more objects are swapped during tracking as they overlap. This is a common issue in Multi-Object Tracking, and some research addresses this problem using appearance-based matching. 

*Then, how well would the MOT models perform in environments where objects are **identical** in appearance with **fast** motion*? 

From this motivation, this project aims to 1) measure the performance of several current MOT works, in conditions where object appearance is completely identical) and 2) identify ways to improve performance. For this experiment, we have chosen Yabawi (the Shell Game), specifically the Matryoshka mini-game from Nintendo Mario Party, as a suitable task. We conduct several experiments based on ByteTrack to seek improvments in performance.



## Team Members

<p>
   <b>:nesting_dolls: Sohyun Yoo (YAI 12th)</b> - Main Experiments (Re-ID, Depth Tracker)</br>
   <b>:nesting_dolls: Jian Kim (YAI 12th)</b> - Custom Dataset Construction</br>
   <b>:nesting_dolls: Junghyun Park (YAI 12th)</b> - Related Works / Main Experiments (Mixed Tracker)</br>
   <b>:nesting_dolls: Gun Jegal (YAI 12th)</b> - Baseline Experiments (FairMOT)</br>
   <b>:nesting_dolls: Kyunghoon Jung (YAI 12th)</b> - Yolov8 Training / Main Experiments (LSTM Tracker) </br>
   <b>:nesting_dolls: Jimin Lee (YAI 11th)</b> - Baseline Experiments (MOTRv2)</br>
  
</p>


## Dataset
We release our annotated Task Dataset:
- TBA

## Setup
To set up the project, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/cygbbhx/kslr.git

```
### 2. Install Dependencies
We follow the setup from [ByteTrack](https://github.com/ifzhang/ByteTrack?tab=readme-ov-file#installation).

## Running Tracking
To run tracking, use the following command:
```bash
bash ./scripts/run_ours.sh
```
- `--wandb`: run sweep based on the provided configuration. (Note: You should modify the configuration according to experiment settings.)


## Main Results
### Baseline
|model | IDF1 |  IDs  |    MOTA     |
|--------|------|-------|------------|
|DeepSORT|<table><thead></thead><tbody><tr>27.9</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>62</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>56.0</tr></tbody></table>|
|MOTRv2|<table><thead></thead><tbody><tr>21.7</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>43</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>45.1</tr></tbody></table>|
|ByteTrack|<table><thead></thead><tbody><tr>**44.4**</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>**60**</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>**91.3**</tr></tbody></table>|
|SparseTrack|<table><thead></thead><tbody><tr>40.5</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>60</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>91.6</tr></tbody></table>|



### Ours
|method | IDF1 |  IDs  |    MOTA     |
|--------|------|-------|------------|
|ByteTrack|<table><thead></thead><tbody><tr>44.4</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>60</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>91.3</tr></tbody></table>|
|ByteTrack + Cascading|<table><thead></thead><tbody><tr>53.12</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>58</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>91.7</tr></tbody></table>|
|ByteTrack + Cascading + Re-ID|<table><thead></thead><tbody><tr>50.8</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>49</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>86.5</tr></tbody></table>|
|ByteTrack + Cascading + Re-ID + reset KF|<table><thead></thead><tbody><tr>**55.11**</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>**46**</tr></tbody></table>|<table><thead><tr></tr></thead><tbody><tr>**96.12**</tr></tbody></table>|

- Note. We experiment various thresholds for each method and choose the best results.
