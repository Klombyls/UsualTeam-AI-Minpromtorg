import shutil, os, random
from tqdm import tqdm


pathDataSet = './cameras'
pathToSaveData = './dataset'
saveInOnePlace = True
sampleRatio = 80


def GetFileNames(camera):
    files = []
    for file in os.listdir(pathDataSet + '/' + camera):
        temp = {}
        temp['path'] = pathDataSet + '/' + camera
        if file[-3:] == 'txt':
            temp['filename'] = file[:-4]
            files.append(temp)
    return files

def MoveFiles(fileWithoutExtention, path):
    if os.path.isfile(fileWithoutExtention + '.txt'):
        if not os.path.isdir(path + '/labels'):
            os.makedirs(path + '/labels')
        shutil.copy(fileWithoutExtention + '.txt', path + '/labels')
    if os.path.isfile(fileWithoutExtention + '.jpg'):
        if not os.path.isdir(path + '/images'):
            os.makedirs(path + '/images')
        shutil.copy(fileWithoutExtention + '.jpg', path + '/images')
    if os.path.isfile(fileWithoutExtention + '.png'):
        if not os.path.isdir(path + '/images'):
            os.makedirs(path + '/images')
        shutil.copy(fileWithoutExtention + '.png', path + '/images')

def ParseData(dataList: list, dir: str):
    for file in tqdm(dataList):
        path = ''
        if saveInOnePlace:
            path = pathToSaveData + '/' + dir
            if not os.path.isdir(path):
                os.makedirs(path)
            MoveFiles(file['path'] + '/' + file['filename'], path)
        else:
            path = pathToSaveData + '/' + file['path'].replace(pathDataSet + '/', './') + '/' + dir
            if not os.path.isdir(path):
                os.makedirs(path)
            MoveFiles(file['path'] + '/' + file['filename'], path)

def Main():
    files = []

    for camera in os.listdir(pathDataSet):
        if os.path.isdir(pathDataSet + '/' + camera):
            files = files + GetFileNames(camera)
            #pass
    
    random.shuffle(files)
    i = int(-(len(files) / 100 * sampleRatio))
    valList = files[:i]
    trainList = files[i:]

    print(len(trainList))
    print('Parse train dataset')
    ParseData(trainList, 'train')
    print('\n\nParse validation dataset')
    ParseData(valList, 'val')
    


if __name__ == '__main__':
    Main()