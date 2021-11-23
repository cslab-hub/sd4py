package org.vikamine.kernel._examples;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.vikamine.kernel.data.Attribute;
import org.vikamine.kernel.data.NominalAttribute;
import org.vikamine.kernel.data.NumericAttribute;
import org.vikamine.kernel.data.Ontology;
import org.vikamine.kernel.data.discretization.EqualWidthDiscretizer;
import org.vikamine.kernel.subgroup.SG;
import org.vikamine.kernel.subgroup.SGFilters;
import org.vikamine.kernel.subgroup.SGSet;
import org.vikamine.kernel.subgroup.analysis.WeightedCoveringAnalyzer;
import org.vikamine.kernel.subgroup.quality.functions.AdjustedResidualQF;
import org.vikamine.kernel.subgroup.quality.functions.BinomialQF;
import org.vikamine.kernel.subgroup.quality.functions.ChiSquareQF;
import org.vikamine.kernel.subgroup.quality.functions.InformationGainQF;
import org.vikamine.kernel.subgroup.quality.functions.LiftQF;
import org.vikamine.kernel.subgroup.quality.functions.PiatetskyShapiroQF;
import org.vikamine.kernel.subgroup.quality.functions.RelativeGainQF;
import org.vikamine.kernel.subgroup.quality.functions.WRAccQF;
import org.vikamine.kernel.subgroup.search.BSD;
import org.vikamine.kernel.subgroup.search.MiningTask;
import org.vikamine.kernel.subgroup.search.NumericBSD;
import org.vikamine.kernel.subgroup.search.SDBeamSearch;
import org.vikamine.kernel.subgroup.search.SDMap;
import org.vikamine.kernel.subgroup.search.SDMapDisjunctive;
import org.vikamine.kernel.subgroup.search.SDMapNumeric;
import org.vikamine.kernel.subgroup.selectors.DefaultSGSelector;
import org.vikamine.kernel.subgroup.selectors.SGSelector;
import org.vikamine.kernel.subgroup.selectors.SGSelectorGenerator;
import org.vikamine.kernel.subgroup.selectors.SelectorGeneratorUtils;
import org.vikamine.kernel.subgroup.target.NumericTarget;
import org.vikamine.kernel.subgroup.target.SGTarget;
import org.vikamine.kernel.subgroup.target.SelectorTarget;

public class PythonDiscoverSubgroups {
    
    public static SGSet discoverSubgroups(
	    Ontology ontology,
	    String target,
	    Set<String> includedAttributes, 
	    //discretise=True,
	    int nbins,
	    String method,
	    String qf,
	    int k,
	    double minQual,
	    int minSize,
	    int minTP,
	    int maxSelectors,
	    boolean ignoreDefaults,
	    boolean filterIrrelevant,
	    String postfilter,
	    Double postfilterParam) {
	
	// Set target value and choose its type
	Attribute targetAttribute = ontology.getAttribute(target);

	SGTarget sgTarget;
	if (targetAttribute.isNominal()) {
	    NominalAttribute nominalTargetAttribute = (NominalAttribute) targetAttribute;
	    // Construct the target: We choose the class + the value at index 0
	    SGSelector targetSelector = new DefaultSGSelector(
		    nominalTargetAttribute,
		    nominalTargetAttribute.getNominalValue(0));
	    sgTarget = new SelectorTarget(targetSelector);
	} else if (targetAttribute.isNumeric()) {
	    NumericAttribute numericTargetAttribute = (NumericAttribute) targetAttribute;
	    sgTarget = new NumericTarget(numericTargetAttribute);
	} else {
	    throw new IllegalStateException("Unknown type of target attribute");
	}
	
	// Filter out attributes that we won't use
	Set<Attribute> setAttributes = ontology.getAttributes();
	for (Iterator<Attribute> iter = setAttributes.iterator(); iter.hasNext();) {
	    Attribute next = iter.next();
	    if (sgTarget.getAttributes().contains(next))
		iter.remove();
	    if (!includedAttributes.contains(next.getDescription())) {
		iter.remove();
	    }
	}
	
	// Generate attribute selectors and add them to a new MiningTask
	SGSelectorGenerator generator = new SGSelectorGenerator.SplitSelectorGenerator(
		new SGSelectorGenerator.SimpleValueSelectorGenerator(),
		new SGSelectorGenerator.SimpleNumericSelectorGenerator(
			new EqualWidthDiscretizer(nbins)));
	List<SGSelector> allSelectors = SelectorGeneratorUtils
		.generateSelectors(generator, setAttributes, ontology.getDataView());
	
	MiningTask task = new MiningTask();
	task.setOntology(ontology);
	List<SGSelector> relevantSelectors = new ArrayList<SGSelector>(
		allSelectors);
	task.setSearchSpace(relevantSelectors);

	// Set the target for the task
	task.setTarget(sgTarget);
	
	// set subgroup to begin search in: here we start with an empty sg
	SG initialSG = new SG(ontology.getDataView(), task.getTarget());
	initialSG.createStatistics(null);
	task.setInitialSG(initialSG);

	// set the quality function:
	switch(qf) {
	case "ares":
	    task.setQualityFunction(new AdjustedResidualQF());
	    break;
	case "bin":
	    task.setQualityFunction(new BinomialQF());
	    break;
	case "chi2":
	    task.setQualityFunction(new ChiSquareQF());
	    break;
	case "gain":
	    task.setQualityFunction(new InformationGainQF());
	    break;
	case "lift":
	    task.setQualityFunction(new LiftQF());
	    break;
	case "ps":
	    task.setQualityFunction(new PiatetskyShapiroQF());
	    break;
	case "relgain":
	    task.setQualityFunction(new RelativeGainQF());
	    break;
	case "wracc":
	    task.setQualityFunction(new WRAccQF());
	    break;
        default:
            throw new IllegalArgumentException("Invalid quality function type. Please select one of: " +
        	    "Adjusted Residuals ares, Binomial Test bin, Chi-Square Test " + 
                    "chi2, Gain gain, Lift lift, Piatetsky-Shapiro ps, Relative Gain relgain, Weighted Relative " + 
                    "Accuracy wracc."
        	    );
	}

	// set the search algorithm
	switch(method) {
	case "bsd":
		if (sgTarget.isNumeric()) {
		    // in the numeric case, we choose the numeric variant of the BSD
		    // algorithm
		    task.setMethodType(NumericBSD.class);
		} else {
		    task.setMethodType(BSD.class);
		}
		break;
	case "sdmap":
		if (sgTarget.isNumeric()) {
		    // in the numeric case, we choose the numeric variant of the BSD
		    // algorithm
		    task.setMethodType(SDMapNumeric.class);
		} else {
		    task.setMethodType(SDMap.class);
		}
		break;
	case "sdmap-dis":
	    	task.setMethodType(SDMapDisjunctive.class);
	    	break;   
	case "beam":
	    	task.setMethodType(SDBeamSearch.class);
	    	break;
        default:
            	throw new IllegalArgumentException("Invalid method type. Please select one of: bsd, sdmap, sdmap-dis, beam");
	}

	// set constraints, e.g.:
	task.setMaxSGDSize(maxSelectors);
	task.setMinSubgroupSize(minSize);
	task.setMaxSGCount(k);
	task.setMinQualityLimit(minQual);
	task.setMinTPSupportAbsolute(minTP);
	task.setIgnoreDefaultValues(ignoreDefaults);
	task.setSuppressStrictlyIrrelevantSubgroups(filterIrrelevant);

	// execute task
	SGSet result = task.performSubgroupDiscovery();

	// Apply filters
	switch(postfilter) {
	case "min_improve_global":
	    result = new SGFilters.MinImprovementFilterGlobal(postfilterParam)
		.filterSGs(result);
	    break;
	case "min_improve_set":
	    result = new SGFilters.MinImprovementFilterOnSGSet(postfilterParam)
		.filterSGs(result);
	    break;
	case "relevancy":
	    result = new SGFilters.RelevancyFilter()
		.filterSGs(result);
	    break;
	case "sig_improve_global":
	    result = new SGFilters.SignificantImprovementFilterGlobal(postfilterParam)
		.filterSGs(result);
	    break;
	case "sig_improve_set":
	    result = new SGFilters.SignificantImprovementFilterOnSet(postfilterParam)
		.filterSGs(result);
	    break;
	case "weighted_covering":
	    result = new WeightedCoveringAnalyzer().getKBestCoveringSubgroups(postfilterParam.intValue(), result, ontology.getDataView(), task.getQualityFunction());
	    break;
	case "":
	    break;
        default:
            throw new IllegalArgumentException("Invalid postfiltering type. Please select one of: " +
        	    "min_improve_global, min_improve_set, relevancy, sig_improve_global, sig_improve_set, weighted_covering");
	}
	
	//List<SG> resultList = result.toSortedList(false);
	
	return result;
	
    }

}
