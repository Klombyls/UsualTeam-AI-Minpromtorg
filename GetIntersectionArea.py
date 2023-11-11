from shapely.geometry import Polygon


class GetIntersectionArea():
    __dangerAreas = {}

    def AddDangerAreas(self, zones: list):
        self.__dangerAreas = zones

    def GetZoneEntryPercentage(self, points: list, cameraName: str):
        result = []
        p = Polygon(points)
        if self.__dangerAreas.get(cameraName):
            for item in self.__dangerAreas[cameraName]:
                x = Polygon(item).intersection(p)
                result.append(x.area / p.area * 100)
            return result