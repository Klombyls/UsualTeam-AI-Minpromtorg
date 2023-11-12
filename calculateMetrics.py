from predict import DangerZoneHandler
import matplotlib.pyplot as plt
import os, cv2, csv, json
from GetIntersectionArea import GetIntersectionArea


class calculatorMetrics():
    def __init__(self, modelName, dangerZonesPath):
        self.checker = DangerZoneHandler(modelName)
        self.checker.loadDangerZone(dangerZonesPath)
        self.dangerZones = {}
        self.intersec = GetIntersectionArea()
        self.loadDangerZone(dangerZonesPath)

    def parseZoneToPolygon(self, zone):
        polygon = []
        for point in zone:
            temp = (point[0], point[1])
            polygon.append(temp)
        return polygon

    def loadDangerZone(self, dangerZonesPath: str,):
        for file in os.listdir(dangerZonesPath):
            if not file[-3:] == 'txt':
                continue
            cameraName = (file[:-4]).replace('danger_', '')
            if cameraName.find('_zone') != -1:
                cameraName = cameraName[:cameraName.find('_zone')]

            js = []
            with open(dangerZonesPath + '/' + file, 'r') as reader:
                js = json.loads('[' + reader.read() + ']')

            if not cameraName in self.dangerZones:
                self.dangerZones[cameraName] = []
            self.dangerZones[cameraName].append(self.parseZoneToPolygon(js))
            self.intersec.AddDangerAreas(self.dangerZones)
        
    def predictFrame(self, file, camera):
        path = file
        img = cv2.imread(path)

        response = self.checker.predictImage(img, camera)
        result = []
        for people in response[:-1][0]:
            result.append(max(people[0]))
        return result

    # coordinates: x, y, w, h          imageSize: w, h
    def parseCoordinateToPolygon(self, coordinates: list, imageSize: list):
        x1 = int(float(imageSize[0]) * (float(coordinates[0]) - float(coordinates[2]) / 2))
        x2 = int(float(imageSize[0]) * (float(coordinates[0]) + float(coordinates[2]) / 2))
        y1 = int(float(imageSize[1]) * (float(coordinates[1]) - float(coordinates[3]) / 2))
        y2 = int(float(imageSize[1]) * (float(coordinates[1]) + float(coordinates[3]) / 2))
        points = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
        return points

    def getRealDataFrame(self, file, cameraName):
        result = []
        img = cv2.imread(file)
        height, width, channels = img.shape
        path = file[:-3] + 'txt'
        coordinates = ""
        f = open(path, 'r')
        for line in f:
            coordinates = line
            coordinates = coordinates.split()
            points = []
            if len(coordinates) != 0:
                coordinates.pop(0)
                points = self.parseCoordinateToPolygon(coordinates, [width, height])
                result.append(max(self.intersec.GetZoneEntryPercentage(points, cameraName)))
        return result

    def calculate(self, file, cameraName):
        predictResult = self.predictFrame(file, cameraName)
        realFrameData = self.getRealDataFrame(file, cameraName)
        predictResult.sort()
        realFrameData.sort()
        if len(predictResult) != len(realFrameData):
            return 0
        for i in range(len(realFrameData)):
            if predictResult[i] >= 15 and realFrameData[i] < 15 or predictResult[i] < 15 and realFrameData[i] >= 15:
                return 0
        return 1

def main():
    modelName = './models/small.pt'
    dangerZonesPath = './danger_zones'
    inputDir = './cameras'
    calculator = calculatorMetrics(modelName, dangerZonesPath)
    result = 0
    i = 0
    for directory in os.listdir(inputDir):
        print(f'Check {directory} directory')
        if not os.path.isdir(inputDir + '/' + directory): continue
        for file in os.listdir(inputDir + '/' + directory):
            if file[-4:] == '.txt': continue
            result = result + calculator.calculate(inputDir + '/' + directory + '/' + file, directory)
            i += 1

    print('\n\n')
    print(f'result = {result} / {i}')
    print('\n\n')

if __name__ == '__main__':
    main()