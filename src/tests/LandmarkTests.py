import sys
import eval.LandmarkHandler


class LandmarkHandlerTests():
  def dirLabTest(self):
    landmarks0 = '/home/fechter/Bilder/DirLab/Case1Pack/ExtremePhases/Case1_300_T00_xyz.txt'
    landmarks5 = '/home/fechter/Bilder/DirLab/Case1Pack/ExtremePhases/Case1_300_T50_xyz.txt'
    pr = eval.LandmarkHandler.PointReader()
    points0 = pr.loadData(landmarks0)
    points5 = pr.loadData(landmarks5)

    pp = eval.LandmarkHandler.PointProcessor()
    pointDistance = pp.calculatePointDistance(points0, points5)
    return True


def main(argv):
  lmht = LandmarkHandlerTests()
  if lmht.dirLabTest():
    print('tests passed')
  else:
    print('tests failed')

if __name__ == '__main__':
    main(sys.argv[1:]) 