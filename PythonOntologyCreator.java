package org.vikamine.kernel._examples;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import org.vikamine.kernel.data.Attribute;
import org.vikamine.kernel.data.AttributeBuilder;
import org.vikamine.kernel.data.DataRecordSet;
import org.vikamine.kernel.data.FullDataRecord;
import org.vikamine.kernel.data.NominalAttribute;
import org.vikamine.kernel.data.Ontology;
import org.vikamine.kernel.data.creators.DataFactory;

public class PythonOntologyCreator {
    
    public Ontology ontology;
    
    private static List<String> unique(String arr[]) {
	
        HashSet<String> set = new HashSet<>();
        
        for (int i=0; i<arr.length; i++)
        {
            if (!set.contains(arr[i]))
            {
                set.add(arr[i]);
            }
        }
        
        List<String> mainList = new ArrayList<String>();
        mainList.addAll(set);
        
        return mainList;
    }
    
    public PythonOntologyCreator(String[] columnNames, String[] columnTypes, double[][] numericArrays, String[][] nominalArrays) throws Exception {
	
	List<Attribute> attributes = new ArrayList<Attribute>();
	AttributeBuilder basicBuilder = new AttributeBuilder();
	
	int colI = 0;
	int nomI = 0;
	int numI = 0;

	for (String colType : columnTypes) {
	    
	    if (colType.equals("numeric")) {
		
		basicBuilder.buildNumericAttribute(columnNames[colI]);
		attributes.add(basicBuilder.getAttribute());	
		
	    }
	    
	    if (colType.equals("nominal")) {
		
		basicBuilder.buildNominalAttribute(columnNames[colI], unique(nominalArrays[nomI]));
		basicBuilder.buildNominalValues();
		attributes.add(basicBuilder.getAttribute());
		
		nomI++;
	    }
	    
	    colI++;
	    
	}
	
	int numberRecords = 0;
	
	if (numericArrays.length != 0) {
	    
	    numberRecords = numericArrays[0].length;
	}
	else {
	    
	    numberRecords = nominalArrays[0].length;
	}
	
	DataRecordSet dataset = new DataRecordSet("dataset", attributes, numberRecords);
	
	for (int rowI = 0; rowI < numberRecords; rowI++) {
	    
	    colI = 0;
	    nomI = 0;
	    numI = 0;
	    
	    double[] record = new double[columnNames.length];
	    
	    for (colI = 0; colI < columnNames.length; colI++) {
		
		if(columnTypes[colI].equals("numeric")) {
		    
		    record[colI] = Double.valueOf(numericArrays[numI][rowI]).doubleValue();
		    numI++;
		}
		
		if(columnTypes[colI].equals("nominal")) {
		    
		    record[colI] = ((NominalAttribute) dataset
			    .getAttribute(columnNames[colI]))
			    .getIndexOfValue(nominalArrays[nomI][rowI]);
		    nomI++;
		}		
		
	    }
	    
	    dataset.add(new FullDataRecord(1, record));
	    
	}
	
	dataset.compactify();
	
	ontology = DataFactory.createOntology(dataset);
	
    }

}