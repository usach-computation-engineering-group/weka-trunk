package weka.classifiers.elm;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import org.apache.commons.math4.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.util.FastMath;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrices;
import weka.classifiers.AbstractClassifier;
import weka.core.BatchPredictor;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

public class WELM extends AbstractClassifier implements BatchPredictor, OptionHandler, TechnicalInformationHandler {
	/**
	 * 
	 */
	private static final long serialVersionUID = 3710237244094711506L;
	/**
	 * Author: Isaac Silva, on 2019/05/26
	 */
	
	private int numberofHiddenNeurons = 60;
	private int distanceTrainingTradeoff = 16;
	
	// Shared between Training and Testing stages.
	private DenseVector bias;
	private DenseMatrix randomHiddenInputW;
	private DenseMatrix trainingOutputWeights;

	/**
	 * Main method for running this class, standalone.
	 */
	public static void main(String[] args) {
		runClassifier(new WELM(), args);
	}

	/**
	 * Returns default capabilities of the classifier.
	 * 
	 * @return the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// Attributes
		result.enable(Capability.NUMERIC_ATTRIBUTES); // Only numeric attributes.

		// Class
		result.enable(Capability.BINARY_CLASS); // Only Binary Classification. NO regression.

		// Instances
		result.enable(Capability.MISSING_VALUES); // Values will be replaced by the mean value.

		return result;
	}

	// Theory behind this classifier
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result = new TechnicalInformation(Type.MASTERSTHESIS);
		result.setValue(Field.AUTHOR, "Weiwei Zong, Guang-Bin Huang, Yiqiang Chen");
		result.setValue(Field.YEAR, "2012");
		result.setValue(Field.TITLE, "Weighted extreme learning machine for imbalance learning");
		result.setValue(Field.PUBLISHER, "Elsevier");
		result.setValue(Field.HTTP, "http://dx.doi.org/10.1016/j.neucom.2012.08.010");
		return result;
	}

	/**
	 * Generates a classifier. Must initialize all fields of the classifier that are
	 * not being set via options (ie. multiple calls of buildClassifier must always
	 * lead to the same result). Must not change the dataset in any way.
	 *
	 * @param data set of instances serving as training data
	 * @exception Exception if the classifier has not been generated successfully
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		// The input of buildClassifier will be the TRAIN Dataset
		
		Instances data = new Instances(instances); // We should not touch input data.
		
		// Next line will return Zero if mayority class is the first one
		// Otherwise, One (Second one).
		// Remember, ** This is a Binary Classifier! ** (At the moment).
		double mode = data.meanOrMode(data.classAttribute());			
		
		// Let's set the Weighting W and binarize the class "attribute" into two binary columns.
		double[] classValues = data.attributeToDoubleArray(data.classIndex());
		double[][] binaryClasses = new double[classValues.length][2]; // First column will be always the * minority * class prediction
		DenseMatrix weightingMatrix = Matrices.identity(classValues.length);
		
		for(int j = 0; j < classValues.length; j++) {
			if(classValues[j] != mode) {
				weightingMatrix.set(j, j, 1.0 / (1.0 * classValues.length));
				binaryClasses[j][0] = 1.0;
				binaryClasses[j][1] = 0.0;
			} else {
				// We will penalize value, as it's above the AVG, it's the mode
				weightingMatrix.set(j, j, 0.618 / (1.0 * classValues.length));
				binaryClasses[j][0] = 0.0;
				binaryClasses[j][1] = 1.0;
			}				
		}
				
		// First, let's generate the shared Bias Vector, for Training and Testing stages.
		this.bias = (DenseVector) Matrices.random(this.numberofHiddenNeurons);

		// Generate the Random Hidden Input Weights, with values between -1 and 1
		// This Weight is shared between Training and Testing.
		this.randomHiddenInputW = Matrices.normalizeBetweenMinusOnetoOne((DenseMatrix) Matrices.random(this.numberofHiddenNeurons, data.numAttributes() - 1)); // No class.
		// Get pre-activation H for training, with the Random Bias
		DenseMatrix normalizedTrainData = Matrices.normalizeBetweenMinusOnetoOne(instances2DenseMatrix(data));
		DenseMatrix preActHtrain = (DenseMatrix) this.randomHiddenInputW.transBmultAdd(normalizedTrainData, Matrices.repeatVector(data.numInstances(), this.bias));
		
		Matrices.sigmoidActivatedH(preActHtrain); // Training.
		this.trainingOutputWeights = getTrainingOutputWeights(preActHtrain, weightingMatrix, new DenseMatrix(binaryClasses));
	}

	/**
	 * Batch prediction method. This default implementation simply calls
	 * distributionForInstance() for each instance in the batch. If subclasses can
	 * produce **BATCH PREDICTIONS** in a more efficient manner than this they should
	 * override this method and also return true from
	 * implementsMoreEfficientBatchPrediction()
	 * 
	 * @param batch the instances to get predictions for
	 * @return an array of probability distributions, one for each instance in the
	 *         batch
	 * @throws Exception if a problem occurs.
	 */
	@Override
	public boolean implementsMoreEfficientBatchPrediction() {
		return true;
	}

	@Override
	public double[][] distributionsForInstances(Instances instances) throws Exception {
		// The input of distributionsForInstances will be the TEST Dataset
		
		Instances data = new Instances(instances); // We should not touch input data.
		
		DenseMatrix normalizedTestData = Matrices.normalizeBetweenMinusOnetoOne(instances2DenseMatrix(data));		
		DenseMatrix preActHtest = (DenseMatrix) this.randomHiddenInputW.transBmultAdd(normalizedTestData, Matrices.repeatVector(data.numInstances(), this.bias));
		
		Matrices.sigmoidActivatedH(preActHtest); // Training.
		
		DenseMatrix results = new DenseMatrix(preActHtest.numColumns(), this.trainingOutputWeights.numColumns());
		preActHtest.transAmult(this.trainingOutputWeights, results);		
		
		// Final output
		double[][] doubleResults = Matrices.getArray(results);
		for (int i = 0; i < preActHtest.numColumns(); i++) {
			if (doubleResults[i][0] > doubleResults[i][1]) { // This only works on Binary classifiers! 
				doubleResults[i][0] = 1.0;
				doubleResults[i][1] = 0.0;
			} else {
				doubleResults[i][0] = 0.0;
				doubleResults[i][1] = 1.0;
			}
		}
				
		return doubleResults;
	}

	/**
	 * Returns a description of the classifier.
	 * 
	 * @return a description of the classifier as a string.
	 */
	@Override
	public String toString() {
		return "Weighted Extreme Learning Machine (WELM)";
	}

	/**
	 * Returns a string describing classifier
	 * 
	 * @return a description suitable for displaying in the explorer/experimenter
	 *         gui
	 */
	public String globalInfo() {
		return "Weighted Extreme Learning Machine, designed for imbalanced input datasets."
				+ getTechnicalInformation().toString();
	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>(14);

		newVector.addElement(new Option(
				"\t Number of Input Hidden Neurons. Default: 20. \n" + "\t Higher values brings better predictions, at cost of speed.",
				"N", 1, "-N (Number of Input Hidden Neurons)"));
		newVector.addElement(new Option(
				"\t Training Error - Marginal Distance trade-off index. Default: 0 (Unitary). \n" + "\t Higher values bring less Training Errors, " + 
						"but more Marginal Distance between predicted classes.",
				"C", 1, "-C (Amount of Training / Distance trade-off)"));

		newVector.addAll(Collections.list(super.listOptions()));

		return newVector.elements();
	}

	/**
	 * 
	 */
	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);

		String numberofHiddenNeuronsString = Utils.getOption('N', options);
		String distanceTrainingTradeoffString = Utils.getOption('C', options);
		
		if (numberofHiddenNeuronsString.length() != 0) {
			this.numberofHiddenNeurons = Integer.parseInt(numberofHiddenNeuronsString);
		} else {
			this.numberofHiddenNeurons = 60;		
		}
		if (distanceTrainingTradeoffString.length() != 0) {
			this.distanceTrainingTradeoff = Integer.parseInt(distanceTrainingTradeoffString);
		} else {
			this.distanceTrainingTradeoff = 16;
		}
		
		Utils.checkForRemainingOptions(options);
	}

	/**
	 * 
	 */
	@Override
	public String[] getOptions() {
		Vector<String> options = new Vector<String>();
		options.add("-N");
		options.add("" + this.numberofHiddenNeurons);
		options.add("-C");
		options.add("" + this.distanceTrainingTradeoff);
		Collections.addAll(options, super.getOptions());
		return options.toArray(new String[0]);
	}
	
	// Helper
	public DenseMatrix instances2DenseMatrix(Instances data) {
		Array2DRowRealMatrix rm = new Array2DRowRealMatrix(data.numInstances(), data.numAttributes() - 1); // Minus class.		
		for (int i = 0; i < data.numAttributes(); i++) {
			if(data.classIndex() != i)
				rm.setColumn(i, data.attributeToDoubleArray(i));
			else
				continue;
		}		
		return new DenseMatrix(rm.getData());
	}
	
	// Solver
    public DenseMatrix getTrainingOutputWeights(DenseMatrix preActH, DenseMatrix weightingMatrix, DenseMatrix trainingClasses) {
    	DenseMatrix hwhTplusCost = (DenseMatrix) Matrices.identity(this.numberofHiddenNeurons).scale(1.0/FastMath.pow(2.0, this.distanceTrainingTradeoff));
    	DenseMatrix hw = new DenseMatrix(preActH.numRows(), preActH.numColumns());
    	preActH.mult(weightingMatrix, hw);
    	DenseMatrix hwhT = new DenseMatrix(hw.numRows(), preActH.numRows());
    	hw.transBmult(preActH, hwhT);
    	hwhTplusCost.add(hwhT);
    	
    	DenseMatrix hwtT = new DenseMatrix(hw.numRows(), trainingClasses.numColumns());
    	hw.mult(trainingClasses, hwtT);
        
    	DenseMatrix solvedMatrix = new DenseMatrix(hwhT.numColumns(), hwtT.numColumns()); 
    	hwhTplusCost.solve(hwtT, solvedMatrix);
    	
    	return solvedMatrix;
    }

	public int getNumberofHiddenNeurons() {
		return numberofHiddenNeurons;
	}

	public void setNumberofHiddenNeurons(int numberofHiddenNeurons) {
		this.numberofHiddenNeurons = numberofHiddenNeurons;
	}

	public int getDistanceTrainingTradeoff() {
		return distanceTrainingTradeoff;
	}

	public void setDistanceTrainingTradeoff(int distanceTrainingTradeoff) {
		this.distanceTrainingTradeoff = distanceTrainingTradeoff;
	}
}
