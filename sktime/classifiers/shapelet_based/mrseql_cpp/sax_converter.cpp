/*
 * sax_converter.cpp
 *
 *  Created on: 27 Jun 2016
 *      Author: LE NGUYEN THACH
 */

#include "sax_converter.h"
#include "common.h"

#include <stdlib.h>
#include <fstream>
#include <unistd.h>
#include <cstring>
#include <algorithm>


using namespace std;



bool convert_timeseries_to_multi_sax(string input_data,string output_sax,int min_ws, int max_ws, int wl, int as ){


	string del = ",";

	std::ifstream infile (input_data);
	std::ofstream outfile (output_sax);



	string line;
	size_t pos = 0;
	double label;
	string sax_str;

	// read data
	vector<string> ts;
	vector<string> y;
	if (infile.is_open()){
		while (getline (infile, line)){
			pos = line.find(del);
			y.push_back(line.substr(0, pos).c_str());
			line.erase(0, pos + del.length());
			ts.push_back(line);
		}
		infile.close();
	} else {
		std::cout << "Invalid input." << std::endl;
		return false;
	}

	if (!outfile.is_open()){
		std::cout << "Invalid output." << std::endl;
		return false;
	}

	//convert to sax
	int config = 0;
	for (int ws = min_ws; ws < max_ws; ws += sqrt(max_ws)){
		cout << config << " " <<  ws << " " <<  wl << " " <<  as << endl;
		SAX sax_converter(ws,wl,as,2);

		for (int i = 0; i < y.size(); i++){
			outfile << config << " " << y[i];
			vector<double> nmts = string_to_double_vector(ts[i],del);
			if (ws < nmts.size()){
				for (string w: sax_converter.timeseries2SAX(nmts)){
					outfile << " " << w;
				}
			}
			outfile << endl;
//			outfile << config << " " << y[i] << " " << sax_converter.timeseries2SAX(ts[i],del) << endl;
		}
		config++;
	}
	outfile.close();

	return true;
}



bool convert_timeseries_to_multi_sax(string input_data,string output_sax, int wl, int as ){


	vector<vector<double>> tss;


	int min_length = 0;

	string del = ",";

	std::ifstream infile (input_data);
	std::ofstream outfile (output_sax);



	string line;
	size_t pos = 0;
	double label;
	string sax_str;

	// read data
	//vector<string> ts;cd
	vector<string> y;
	if (infile.is_open()){
		while (getline (infile, line)){
			pos = line.find(del);
			y.push_back(line.substr(0, pos).c_str());
			line.erase(0, pos + del.length());
			//ts.push_back(line);
			tss.push_back(string_to_double_vector(line,del));
			if (min_length == 0){
				min_length = tss.back().size();
			} else {
				min_length = !(min_length < tss.back().size()) ? tss.back().size() : min_length; // min 2 numbers
			}
		}
		infile.close();
	} else {
		std::cout << "Invalid input." << std::endl;
		return false;
	}

	if (!outfile.is_open()){
		std::cout << "Invalid output." << std::endl;
		return false;
	}

	//cout << "Min length of data:" << min_length << endl;
	//convert to sax
	int config = 0;
	for (int ws = 16; ws < min_length; ws += sqrt(min_length)){
		cout << config << " " <<  ws << " " <<  wl << " " <<  as << endl;
		//int ws = int(0.2*min_length);
		SAX sax_converter(ws,wl,as,2);
		for (int i = 0; i < y.size(); i++){
			outfile << config << " " << y[i];
			for (string word : sax_converter.timeseries2SAX(tss[i])){
				outfile << " " << word;
			}
			outfile << endl;
		}
		config++;
	}

	outfile.close();

	return true;
}

bool find_patterns(string ts_file,string pt_file,string output){
	vector<vector<double>> tss;
	vector<vector<double>> accu_scores;
	vector<string> labels;

	int min_length = 0;

	string ts_del = ",";
	string del = ",";

	std::ifstream tsfile (ts_file);
	std::ifstream ptfile (pt_file);
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

	if (ptfile.is_open()){
		while (getline (ptfile, line)){
			pos = line.find(del);
			int window_size = atoi(line.substr(0, pos).c_str());
			line.erase(0, pos + del.length());

			pos = line.find(del);
			int word_length = atoi(line.substr(0, pos).c_str());
			line.erase(0, pos + del.length());

			pos = line.find(del);
			int alphabet_size = atoi(line.substr(0, pos).c_str());
			line.erase(0, pos + del.length());

			pos = line.find(del);
			double score = atof(line.substr(0, pos).c_str());
			line.erase(0, pos + del.length());

			SAX sax_converter(window_size, word_length, alphabet_size, 2);
			cout << "Looking for sequence " << line << " of score " << score << " with config [" << window_size << "," << word_length << "," << alphabet_size << "]" << endl;
			for (int i = 0; i < tss.size();i++){
				//sax_converter.detect_patterns(tss[i],line,score,accu_scores[i]);
				sax_converter.detect_patterns_and_normalize_score(tss[i],line,score,accu_scores[i]);
			}

			//ts.push_back(line);

		}

		ptfile.close();
	} else {
		std::cout << "Invalid patterns input." << std::endl;
		return false;
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


	int min_ws = 20;
	int max_ws = 100;
	int word_length = 16;
	int alphabet = 4;

	// character-level 1
	// word-level 0
	int token_type = 0;

	int reduction_strategy = 2;


	string input;
	string output;
	string ptf;

	int mode = 0; // auto detect length of time series


	int opt;
	while ((opt = getopt(argc, argv, "i:o:n:N:w:a:m:p:")) != -1) {
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
		case 'n':
			min_ws = atoi(optarg);
			break;
		case 'N':
			max_ws = atoi(optarg);
			break;
		case 'w':
			word_length = atoi(optarg);
			break;
		case 'a':
			alphabet = atoi(optarg);
			break;
		case 'm':
			mode = atoi(optarg);
			break;

		default:
			std::cout << "Usage: " << argv[0] << std::endl;
			return -1;
		}
	}

	if (mode == 0){// auto detect length of time series
		convert_timeseries_to_multi_sax(input,output,word_length,alphabet);
	} else if (mode == 1){// with min and max window defined by user
		convert_timeseries_to_multi_sax(input,output,min_ws,max_ws,word_length,alphabet);
	} else if (mode == 2){ // find sax pattern in time series
		find_patterns(input,ptf,output);
	} //else if (mode == 3) {// convert both train and test

	//}
	//cout << "finish main" << endl;
	return 1;

}
