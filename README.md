# MNET_experiment
## 1. Cài đặt môi trường
```
pip install requirements.txt
```

## 2. Tải file trọng số số model
Bạn hay tải file trọng số model [tại đây](https://drive.google.com/drive/folders/1w54NjX69jYioTY8YYzAQasVhAlfI6x10?usp=drive_link) của tác giả bài báo và đặt nó tại "thư mục weight"

## 3. Chạy để thử nghiệm
```
python inference.py -w weight/model_best.pth.tar -i example_images/COCO_val2014_000000014338-Jules_Logo-175.png
```