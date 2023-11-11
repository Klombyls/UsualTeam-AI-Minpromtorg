from ultralytics import YOLO
import cv2, json, os
import matplotlib.pyplot as plt
from pathlib import Path
from GetIntersectionArea import GetIntersectionArea


modelName = 'best2.pt'
imgpath = './Spp-210-K1-3-3-6.jpg'
imgpath = './Php-Angc-K3-1.jpg'
dangerZonesPath = './danger_zones'

class CheckDungerZone():
    def __init__(self, modelName):
        self.modelName = modelName
        self.model = YOLO(self.modelName)
        self.intersection = GetIntersectionArea()
        self.dangerZones = {}

    def drawDangerZones(self, img, polygons):
        color = (255, 255, 0)
        thickness = 3
        for polygon in polygons:
            for i in reversed(range(len(polygon))):
                cv2.line(img, polygon[i], polygon[i - 1], color, thickness)

    def drawPeople(self, img, people, intersect):
        point1 = (int(people['box']['x1']), int(people['box']['y1']))
        point2 = (int(people['box']['x2']), int(people['box']['y2']))
        pointForText = (int(people['box']['x1']), int(people['box']['y1']) - 5)
        cv2.rectangle(img, point1, point2, (255, 0, 0), 3)
        cv2.putText(img, str(round(people['confidence'], 3)), pointForText, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 2)
        pointForText = (int(people['box']['x1']), int(people['box']['y1']) - 25)
        cv2.putText(img, str(round(max(intersect), 3)), pointForText, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 2)

    def calculatePredict(self, response, img, cameraName):
        result = []
        for recognizedObjects in response:
            js = json.loads(recognizedObjects.tojson())
            for people in js:
                points = [
                    (int(people['box']['x1']), int(people['box']['y1'])),
                    (int(people['box']['x1']), int(people['box']['y2'])),
                    (int(people['box']['x2']), int(people['box']['y2'])),
                    (int(people['box']['x2']), int(people['box']['y1']))
                ]
                intersect = self.intersection.GetZoneEntryPercentage(points, cameraName)
                self.drawPeople(img, people, intersect)
                result.append([intersect, people['confidence']])
        return [result, img]

    def predictFile(self, filepath: str):
        img = cv2.imread(filepath)
        cameraName = Path(filepath).stem
        if  self.dangerZones.get(cameraName):
            self.drawDangerZones(img, self.dangerZones[cameraName])
        response = self.model(filepath)
        return self.calculatePredict(response, img, cameraName)
    
    def predictImage(self, img, cameraName):
        result = []
        if  self.dangerZones.get(cameraName):
            self.drawDangerZones(img, self.dangerZones[cameraName])
        response = self.model(img)
        return self.calculatePredict(response, img, cameraName)
            
    def parseZoneToPolygon(self, zone):
        polygon = []
        for point in zone:
            temp = (point[0], point[1])
            polygon.append(temp)
        return polygon

    def loadDangerZone(self, dangerZonesPath: str):
        for file in os.listdir(dangerZonesPath):
            if file[-3:] == 'txt':
                cameraname = (file[:-4]).replace('danger_', '')
                if cameraname.find('_zone') != -1:
                    cameraname = cameraname[:cameraname.find('_zone')]
                js = []
                with open(dangerZonesPath + '/' + file, 'r') as reader:
                    js = json.loads('[' + reader.read() + ']')
                if not cameraname in self.dangerZones:
                    self.dangerZones[cameraname] = []
                self.dangerZones[cameraname].append(self.parseZoneToPolygon(js))
        self.intersection.AddDangerAreas(self.dangerZones)
        return self.dangerZones


def main():
    checker = CheckDungerZone(modelName)
    checker.loadDangerZone(dangerZonesPath)
    inputDir = './test'
    for file in os.listdir(inputDir):
        path = inputDir + '/' + file
        img = cv2.imread(path)
        camera = Path(path).stem

        result = checker.predictImage(img, camera)
        #result = checker.predictFile(path)

        for people in result[:-1][0]:
            if max(people[0]) > 15:
                print('True', max(people[0]))
            else: print('False', max(people[0]))
        plt.imshow(result[-1])
        plt.show()


if __name__ == '__main__':
    main()