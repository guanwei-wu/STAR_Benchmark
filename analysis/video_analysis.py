import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset_root = "/data/user_data/gdhanuka/STAR_dataset"
video_segments = pd.read_csv(f"{dataset_root}/Video_Segments.csv")

print(video_segments.head())
print(video_segments.shape)

# process the start and end times, and store the lengths of the videos
lengths = []
for i in range(video_segments.shape[0]):
    start = video_segments.iloc[i]["start"]
    end = video_segments.iloc[i]["end"]
    lengths.append(end - start)

# # make plot of video lengths
# sns.histplot(lengths, bins=50, kde=True)
# plt.xlabel("Video Length (s)")
# plt.ylabel("Frequency")
# plt.title("Distribution of Video Lengths")
# plt.savefig("video_lengths.png")

# # print number of videos having length from 0 to 10 seconds
# count = 0
# for length in lengths:
#     if length <= 10:
#         count += 1
# print("Number of videos having length <= 10 seconds:", count)

# count = 0
# for length in lengths:
#     if length >= 20:
#         count += 1

# print("Number of videos having length >= 20 seconds:", count)

# analyse number of segments per video_id
video_id_counts = video_segments["video_id"].value_counts()

# print number of videos having 1 segment
count = 0
for value in video_id_counts:
    if value < 5:
        count += 1
print("Number of videos less than 5 segment:", count)

# print(video_id_counts)

# # make plot of number of segments per video_id
# plt.xlabel("Number of Segments")
# plt.ylabel("Frequency")
# plt.title("Distribution of Number of Segments per Video ID")
# sns.histplot(video_id_counts, bins=50, kde=True)
# plt.savefig("segments_per_video_id.png")
