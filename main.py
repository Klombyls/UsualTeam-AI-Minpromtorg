from predict import DangerZoneHandler
import matplotlib.pyplot as plt
import os, cv2, csv
from pathlib import Path


def main():
    modelName = './models/large.pt'
    dangerZonesPath = './danger_zones'
    inputDir = './videos'

    f = open('./result.csv', 'w', newline='')
    writer = csv.writer(f, delimiter = ",")
    writer.writerow(['camera_name'+','+'\"'+"frame_filename"+'\"'+','+'\"'+"in_danger_zone"+'\"'+','+'\"'+"percent"+'\"'])
    checker = DangerZoneHandler(modelName)
    checker.loadDangerZone(dangerZonesPath)
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
                    l = []
                    if max(people[0]) >= 15:
                        writer.writerow([camera+','+'\"'+file+'\"'+','+'\"'+'True'+'\"'+','+'\"'+str(round(max(people[0]), 2))+'\"'])
                    else:
                        writer.writerow([camera+','+'\"'+file+'\"'+','+'\"'+'False'+'\"'+','+'\"'+"0.0"+'\"'])
                plt.imshow(img)
                plt.show()


if __name__ == '__main__':
    main()