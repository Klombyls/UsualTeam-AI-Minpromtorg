import cv2, json
from predict import CheckDungerZone
from ultralytics import YOLO

modelName = 'best2.pt'
dangerZonesPath = './danger_zones'
model = YOLO(modelName)

def drawRectangle(img, polygons):
    color = (255, 255, 0)
    thickness = 3
    for i in reversed(range(len(polygons))):
        cv2.line(img, polygons[i], polygons[i - 1], color, thickness)

def main():
    checker = CheckDungerZone(modelName)
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
        response = model(frame)
        for recognizedObjects in response:
            js = json.loads(recognizedObjects.tojson())
            for object in js:
                points = [
                    (int(object['box']['x1']), int(object['box']['y1'])),
                    (int(object['box']['x1']), int(object['box']['y2'])),
                    (int(object['box']['x2']), int(object['box']['y2'])),
                    (int(object['box']['x2']), int(object['box']['y1']))
                ]
                drawRectangle(frame, points)
        cv2.imshow("Face detection", frame)

def test():
    checker = CheckDungerZone(modelName)
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
        result = checker.predictImage(frame, '')
        cv2.imshow("Face detection", result[-1])

if __name__ == '__main__':
    test()