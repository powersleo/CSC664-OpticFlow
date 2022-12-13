import OpticFlow

cellsSample = input("number of cells to sample:")
frameToSample = input("number of frames to sample:")
frameTime = input("time per frame:")
for i in range(0,int(cellsSample)):
    row = input("row of cell:")
    column = input("column of cell:")
    visual = input("view 0 or 1:")
    if(int(visual) == 1):
        saveFrames = input("save frames 0 or 1:")
    else:
        saveFrames = 0
    OpticFlow.calculateOpticFlowData(frameNumberX=int(row),frameNumberY=int(column),frameTime=int(frameTime),frameToSample=int(frameToSample),writeFrames=bool(saveFrames), gridOffset=10, view=bool(int(visual)), filterType=0, magnitudeConstraint=0, temporalNoiseReduction=False)

    