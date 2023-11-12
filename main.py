from predict import DangerZoneHandler
import matplotlib.pyplot as plt
import os, cv2, csv
from pathlib import Path


def main():
    modelName = './models/small.pt'
    dangerZonesPath = './danger_zones'

    f = open('./result.csv', 'w', newline='')
    writer = csv.writer(f, delimiter = ";")
    checker = DangerZoneHandler(modelName)
    checker.loadDangerZone(dangerZonesPath)
    inputDir = './test'
    for directory in os.listdir(inputDir):
        if os.path.isdir(inputDir + '/' + directory):
            for file in os.listdir(inputDir + '/' + directory):
                if file[-4:] == '.txt':
                    continue
                path = inputDir + '/' + directory + '/' + file
                img = cv2.imread(path)
                camera = directory

                result = checker.predictImage(img, camera)

                for people in result[:-1][0]:
                    if max(people[0]) >= 15:
                        print('True', max(people[0]))
                        writer.writerow([path, 'True', max(people[0])])
                    else: 
                        print('False', max(people[0]))
                        writer.writerow([path, 'False', max(people[0])])
                plt.imshow(img)
                plt.show()


if __name__ == '__main__':
    main()