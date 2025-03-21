# /bin/bash!

mkdir data
cd data

wget https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Question_Answer_SituationGraph/STAR_train.json
wget https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Question_Answer_SituationGraph/STAR_val.json
wget https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Question_Answer_SituationGraph/STAR_test.json
wget https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Question_Answer_SituationGraph/split_file.json

wget https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Templates_Programs/QA_templates.csv
wget https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Templates_Programs/QA_programs.csv

wget https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Situation_Video_Data/Video_Segments.csv
wget https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Situation_Video_Data/Video_Keyframe_IDs.csv

wget https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Annotations/classes.zip
wget https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Annotations/object_bbox_and_relationship.pkl
wget https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Annotations/pose.zip
wget https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Annotations/person_bbox.pkl

unzip classes.zip
rm classes.zip

unzip pose.zip  # very large
rm pose.zip

# get raw video frames
wget https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip

unzip Charades_v1_480.zip
rm Charades_v1_480.zip

# Run the Python script
# For SHG-VQA
python - <<EOF
import gdown

url_val_updated = "https://drive.google.com/uc?export=download&id=1mkFFuzBvb8R6QR9q4dwGoRnMc8GHPdGv"
url_train_updated = "https://drive.google.com/uc?export=download&id=1GGG2AimF9ho8Uj406_GsN1OtL4scimmT"

output_val_updated = 'STAR_val_updated.json'
output_train_updated = 'STAR_train_updated.json'

gdown.download(url_val_updated, output_val_updated, quiet=False)
gdown.download(url_train_updated, output_train_updated, quiet=False)
EOF

python convert_json_to_pkl.py  # generate STAR_train.pkl, STAR_val.pkl

