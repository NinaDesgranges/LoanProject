import data_analysis as da
import accepted_refused_model as acc_ref_model
import rate_model as rm
from settings import DATA
# da.createNewDataset()

# da.createDatasetForAcceptedVsRefused()

# da.getInfo()

# da.templateUsMapPercAcceptedLoan()

# acc_ref_model.preliminaryAnalysis()

# acc_ref_model.gridSearchLogisticRegression(acc_ref_model.dataTransformation())

# acc_ref_model.outputGridSearchLogisticRegression(path=DATA + 'accepted_refused_ds_small.csv')

# acc_ref_model.gridSearchLogisticRegression(acc_ref_model.dataTransformation())

# acc_ref_model.outputGridSearchLogisticRegression() #path=DATA + 'accepted_refused_ds_small.csv')

# da.createSmallDataset()

# da.templateRateCorrelation()

# da.getRegionFromBoundaries()

# da.countRateOverTime()

# da.templateAcceptedLoanPerRegion()

# da.test()

# da.templateRateCorrelation('CA')

# da.templateUsMapPercAcceptedLoan()

# acc_ref_model.outputGridSearchLogisticRegression()

# da.templateROC(c='c002812')

# da.templateROC(c='c00001')

# rm.data_cleaning()

# rm.featuresReduction()

# rm.modelingSVR()

# rm.modelingRandomForest()

# rm.modelingKNN()

# da.templateMSEComparison()

rm.ensambleModel()
# linear regression
# 16.9224284813
# 0.119117474121

# ('kern', Nystroem(kernel='poly')),
# ('linear_reg', svm.LinearSVR())
# 17.611769317
# 0.0832344271216

# da.templateCoefRegression()