import os
os.environ["HF_TOKEN"]="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"
from datasets import load_dataset
from collections import defaultdict
from PIL import Image


# Hugging Face에서 ImageNet 데이터셋 로드
dataset = load_dataset("zh-plus/tiny-imagenet",split='valid')

# 각 클래스별로 이미지를 저장할 디렉토리 생성
output_dir = "imagenet_samples/val/"
os.makedirs(output_dir, exist_ok=True)


# 클래스별 이미지 카운트를 추적하기 위한 딕셔너리
class_count = defaultdict(int)



for item in dataset:

    label = item['label']
    image = item['image']
    
    if class_count[label] < 2:
        # 클래스 이름 얻기
        class_name = dataset.features['label'].int2str(label)
        
        # 클래스별 디렉토리 생성
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # 이미지 저장
        image_path = os.path.join(class_dir, f"{class_name}_{class_count[label]}.jpg")
        image.save(image_path)
        
        class_count[label] += 1
    
#     # 모든 클래스에 대해 2개씩 저장했다면 종료
# if all(count == 2 for count in class_count.values()):
#     break