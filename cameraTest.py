import cv2, json
from predict import DangerZoneHandler
from ultralytics import YOLO

modelName = './models/nano.pt'
dangerZonesPath = './danger_zones'
model = YOLO(modelName)

def drawRectangle(img, polygons):
    color = (255, 255, 0)
    thickness = 3
    for i in reversed(range(len(polygons))):
        cv2.line(img, polygons[i], polygons[i - 1], color, thickness)


def main():
    checker = DangerZoneHandler(modelName)
    checker.loadDangerZone(dangerZonesPath)
    # получаем видео с камеры
    video=cv2.VideoCapture(0)
    # пока не нажата любая клавиша — выполняем цикл
    while cv2.waitKey(1)<0:
        # получаем очередной кадр с камеры
        hasFrame,frame=video.read()
        # если кадра нет
        if not hasFrame:
            # останавливаемся и выходим из цикла
            cv2.waitKey()
            break
        # выводим картинку с камеры
        cv2.resize(frame, (1920, 1080))
        result = checker.predictImage(frame, 'my camera')
        cv2.imshow("Face detection", result[-1])

if __name__ == '__main__':
    main()