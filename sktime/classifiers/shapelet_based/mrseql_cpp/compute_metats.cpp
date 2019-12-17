/*
 * compute_metats.cpp
 *
 *  Created on: 14 Aug 2018
 *      Author: thachln
 */


#include "sax_converter.h"
#include "common.h"

#include <stdlib.h>
#include <fstream>
#include <unistd.h>
#include <cstring>
#include <algorithm>

using namespace std;


class RConfig{
public:
	int dim;
	int window_size;
	int word_length;
	int alphabet_size;
};

class SymbolicFeatures{
public:
	RConfig cf;
	vector<string> sequences;
	vector<double> scores;

	SymbolicFeatures(RConfig cf){
		this->cf = cf;
	}

	SymbolicFeatures(int dim,int window_size,int word_length,int alphabet_size){
		this->cf.dim = dim;
		this->cf.window_size = window_size;
		this->cf.word_length = word_length;
		this->cf.alphabet_size = alphabet_size;
	}

	void addFeature(string seq, double score){
		sequences.push_back(seq);
		scores.push_back(score);
	}
	void print_features(){
		cout << cf.dim << "," << cf.window_size << "," << cf.word_length << "," << cf.alphabet_size << endl;
		for (int i = 0; i < sequences.size(); i++){
			cout << sequences[i] << "," << scores[i] << endl;
		}
	}
};



vector<SymbolicFeatures> read_univariate_features(string config_file, string feature_file){
	vector<SymbolicFeatures> features;
	string del = ",";
	string line;
	std::ifstream configf (config_file);
	if (configf.is_open()){
		while (getline (configf, line)){
			vector<int> vec_desc = string_to_int_vector(line, " ");
			features.push_back(SymbolicFeatures(0, vec_desc[1], vec_desc[2], vec_desc[3]));
		}
	} else {
		std::cout << "Invalid config file." << std::endl;
	}
	configf.close();

	std::ifstream ftf (feature_file);
	if (ftf.is_open()){
		while (getline (ftf, line)){
			size_t pos = 0;

			pos = line.find(del);
			int cf = atoi(line.substr(0, pos).c_str());
			line.erase(0, pos + del.length());

			pos = line.find(del);
			double sc = atof(line.substr(0, pos).c_str());
			line.erase(0, pos + del.length());

			features[cf].addFeature(line,sc);



		}
	} else {
		std::cout << "Invalid feature file." << std::endl;
	}
	ftf.close();

	return features;

}





bool find_patterns(string ts_file, string pt_file, string cf_file, string output){

	// read data

	vector<vector<double>> tss;
	vector<vector<double>> accu_scores;
	vector<string> labels;

	int min_length = 0;

	string ts_del = ",";
	string del = ",";

	std::ifstream tsfile (ts_file);
	std::ofstream outfile (output);



	string line;
	size_t pos = 0;
	double label;
	string sax_str;

	// read data

	if (tsfile.is_open()){
		while (getline (tsfile, line)){
			pos = line.find(ts_del);
			labels.push_back(line.substr(0, pos).c_str());
			line.erase(0, pos + ts_del.length());
			//ts.push_back(line);
			tss.push_back(string_to_double_vector(line,ts_del));
			accu_scores.push_back(vector<double>(tss.back().size(),0.0));
			//accu_scores.back().resize(tss.back().size());
			//std::fill(accu_scores.back().begin(), accu_scores.back().end(), 0.0);
		}
		tsfile.close();
	} else {
		std::cout << "Invalid time series input." << std::endl;
		return false;
	}

	vector<SymbolicFeatures> features = read_univariate_features(cf_file,pt_file);

	for(SymbolicFeatures fts:features){
		fts.print_features();
		SAX sax_converter(fts.cf.window_size, fts.cf.word_length, fts.cf.alphabet_size, 2);
		for (int i = 0; i < tss.size();i++){
			sax_converter.detect_multiple_patterns(tss[i],fts.sequences,fts.scores,accu_scores[i]);
		}
	}

	if (!outfile.is_open()){
		std::cout << "Invalid output." << std::endl;
		return false;
	} else {
		for (vector<double> ts_sc:accu_scores){
			if (ts_sc.size() > 0){
				outfile << ts_sc[0];
			}
			for (int i = 1; i < ts_sc.size();i++){
				outfile << "," << ts_sc[i];
			}
			outfile << endl;
		}
		outfile.close();
	}

	return true;
}


int main(int argc, char **argv){



	string input;
	string output;
	string ptf;
	string config;


	int opt;
	while ((opt = getopt(argc, argv, "i:o:p:c:")) != -1) {
		switch(opt) {
		case 'i':
			input = string (optarg);
			break;
		case 'o':
			output = string(optarg);
			break;
		case 'p':
			ptf = string(optarg);
			break;
		case 'c':
			config = string(optarg);
			break;



		default:
			std::cout << "Usage: " << argv[0] << std::endl;
			return -1;
		}
	}


	find_patterns(input,ptf,config,output);

	return 1;

}
