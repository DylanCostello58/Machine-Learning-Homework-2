from ultralytics import YOLO

def get_yolo_model(num_classes):
    model = YOLO("yolov8n.pt") # pretrained YOLOv8n
    model.model[-1].nc = num_classes
    model.model[-1].names = [str(i) for i in range(num_classes)]
    return model
