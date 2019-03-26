class InfereMonitor(object):

    def __init__(self, shape, numLabels):
        self.shape = shape
        self.numLabels = numLabels
        self.imgplot = []

    def appendLabelVector(self, labelVector):
        # save the labels at each iteration, to examine later.
        labelVector = labelVector.reshape(self.shape)
        self.imgplot.append([labelVector])

    def checkEnergy(self, inference):
        gm = inference.gm()
        # the arg method returns the (class) labeling at each pixel.
        labelVector = inference.arg()
        # evaluate the energy of the graph given the current labeling.
        print('energy %s' % gm.evaluate(labelVector))
        self.appendLabelVector(labelVector)

    def begin(self, inference):
        print('beginning of inference')
        self.checkEnergy(inference)

    def end(self, inference):
        print('end of inference')

    def visit(self, inference):
        self.checkEnergy(inference)
